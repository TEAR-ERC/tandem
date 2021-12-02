#include "SeasQDDiscreteGreenOperator.h"
#include "common/PetscUtil.h"

#include "form/RefElement.h"
#include "parallel/LocalGhostCompositeView.h"
#include "util/Stopwatch.h"

namespace tndm {

SeasQDDiscreteGreenOperator::SeasQDDiscreteGreenOperator(
    std::unique_ptr<typename base::dg_t> dgop, std::unique_ptr<AbstractAdapterOperator> adapter,
    std::unique_ptr<AbstractFrictionOperator> friction, bool matrix_free, MGConfig const& mg_config)
    : base(std::move(dgop), std::move(adapter), std::move(friction), matrix_free, mg_config) {
    //compute_discrete_greens_function();
    get_discrete_greens_function();
}

SeasQDDiscreteGreenOperator::~SeasQDDiscreteGreenOperator() { MatDestroy(&G_); }

void SeasQDDiscreteGreenOperator::set_boundary(
    std::unique_ptr<AbstractFacetFunctionalFactory> fun) {
    base::set_boundary(std::move(fun));
   //compute_boundary_traction();
   get_boundary_traction();
}

void SeasQDDiscreteGreenOperator::update_internal_state(double time, BlockVector const& state,
                                                        bool state_changed_since_last_rhs,
                                                        bool require_traction,
                                                        bool require_displacement) {
    bool require_solve = state_changed_since_last_rhs && require_traction;
    bool require_solve_domain = require_displacement;
    if (!require_solve && !require_solve_domain) {
        return;
    }

    if (require_solve) {
        update_traction(time, state);
    }

    if (require_solve_domain) {
        base::update_ghost_state(state);
        base::solve(time, base::make_state_view(state));
    }
}

void SeasQDDiscreteGreenOperator::update_traction(double time, BlockVector const& state) {
    base::update_ghost_state(state);
    auto state_view = base::make_state_view(state);
    for (std::size_t faultNo = 0, num = base::friction().num_local_elements(); faultNo < num;
         ++faultNo) {
        S_->insert_block(faultNo, state_view.get_block(faultNo));
    }
    S_->begin_assembly();
    S_->end_assembly();

    CHKERRTHROW(MatMult(G_, S_->vec(), base::traction_.vec()));
    CHKERRTHROW(VecAXPY(base::traction_.vec(), time, t_boundary_->vec()));
}

void SeasQDDiscreteGreenOperator::compute_discrete_greens_function() {
    auto slip_block_size = base::friction().slip_block_size();

    PetscInt num_local_elements = base::adapter().num_local_elements();
    PetscInt m_bs = base::adapter().traction_block_size();
    PetscInt n_bs = 1;
    PetscInt m = num_local_elements * m_bs;
    PetscInt n = num_local_elements * slip_block_size * n_bs;

    MPI_Comm comm = base::comm();

    int rank;
    MPI_Comm_rank(comm, &rank);

    PetscInt mb_offset = 0;
    PetscInt nb_offset = 0;
    MPI_Scan(&num_local_elements, &mb_offset, 1, MPIU_INT, MPI_SUM, comm);
    mb_offset -= num_local_elements;
    MPI_Scan(&n, &nb_offset, 1, MPIU_INT, MPI_SUM, comm);
    nb_offset -= n;

    CHKERRTHROW(MatCreateDense(comm, m, n, PETSC_DECIDE, PETSC_DECIDE, nullptr, &G_));
    CHKERRTHROW(MatSetBlockSizes(G_, m_bs, n_bs));

    S_ = std::make_unique<PetscVector>(slip_block_size, num_local_elements, comm);
    t_boundary_ = std::make_unique<PetscVector>(m_bs, num_local_elements, comm);

    auto scatter = Scatter(base::adapter().fault_map().scatter_plan());
    auto ghost = scatter.template recv_prototype<double>(slip_block_size, ALIGNMENT);

    PetscInt N;
    CHKERRTHROW(VecGetSize(S_->vec(), &N));

    Stopwatch sw;
    double solve_time = 0.0;
    for (PetscInt i = 0; i < N; ++i) {

        if (rank == 0) {
            std::cout << "Computing Green's function " << (i + 1) << "/" << N;
        }
        sw.start();
        CHKERRTHROW(VecZeroEntries(S_->vec()));
        if (i >= nb_offset && i < nb_offset + m) {
            PetscScalar one = 1.0;
            CHKERRTHROW(VecSetValue(S_->vec(), i, one, INSERT_VALUES));
        }
        S_->begin_assembly();
        S_->end_assembly();

        scatter.begin_scatter(*S_, ghost);
        scatter.wait_scatter();

        auto S_view = LocalGhostCompositeView(*S_, ghost);
        base::solve(0.0, S_view);
        base::update_traction(S_view);

        auto traction_handle = base::traction_.begin_access_readonly();
        for (std::size_t faultNo = 0; faultNo < num_local_elements; ++faultNo) {
            PetscInt g_m = mb_offset + faultNo;
            PetscInt g_n = i;
            auto traction_block = traction_handle.subtensor(slice{}, faultNo);
            CHKERRTHROW(
                MatSetValuesBlocked(G_, 1, &g_m, 1, &g_n, traction_block.data(), INSERT_VALUES));
        }
        base::traction_.end_access_readonly(traction_handle);
        solve_time += sw.stop();
        if (rank == 0) {
            constexpr double Days = 3600.0 * 24.0;
            constexpr double Hours = 3600.0;
            constexpr double Minutes = 60.0;
            double avg_time = solve_time / (i + 1);
            double etl = avg_time * (N - i - 1);
            double etl_d = std::floor(etl / Days);
            etl -= etl_d * Days;
            double etl_h = std::floor(etl / Hours);
            etl -= etl_h * Hours;
            double etl_m = std::floor(etl / Minutes);
            etl -= etl_m * Minutes;
            std::cout << " (" << etl_d << "d " << etl_h << "h " << etl_m << "m " << std::floor(etl)
                      << "s left)" << std::endl;
        }
    }

    CHKERRTHROW(MatAssemblyBegin(G_, MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd(G_, MAT_FINAL_ASSEMBLY));
}

void SeasQDDiscreteGreenOperator::compute_boundary_traction() {
    MPI_Comm comm = base::comm();
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        std::cout << "Computing boundary Green's function" << std::endl;
    }

    auto slip_block_size = base::friction().slip_block_size();
    auto scatter = Scatter(base::adapter().fault_map().scatter_plan());
    auto ghost = scatter.template recv_prototype<double>(slip_block_size, ALIGNMENT);

    CHKERRTHROW(VecZeroEntries(S_->vec()));
    scatter.begin_scatter(*S_, ghost);
    scatter.wait_scatter();

    auto S_view = LocalGhostCompositeView(*S_, ghost);
    base::solve(1.0, S_view);
    base::update_traction(S_view);

    CHKERRTHROW(VecCopy(base::traction_.vec(), t_boundary_->vec()));
}

void SeasQDDiscreteGreenOperator::create_discrete_greens_function() {
  auto slip_block_size = base::friction().slip_block_size();
  
  PetscInt num_local_elements = base::adapter().num_local_elements();
  PetscInt m_bs = base::adapter().traction_block_size();
  PetscInt n_bs = 1;
  PetscInt m = num_local_elements * m_bs;
  PetscInt n = num_local_elements * slip_block_size * n_bs;
  PetscInt mb_offset = 0;
  PetscInt nb_offset = 0;
  int rank;
  MPI_Comm comm = base::comm();
  
  CHKERRTHROW(PetscPrintf(comm,"create_discrete_greens_function()\n"));
  MPI_Comm_rank(comm, &rank);
  MPI_Scan(&num_local_elements, &mb_offset, 1, MPIU_INT, MPI_SUM, comm);
  mb_offset -= num_local_elements;
  MPI_Scan(&n, &nb_offset, 1, MPIU_INT, MPI_SUM, comm);
  nb_offset -= n;
  
  CHKERRTHROW(MatCreateDense(comm, m, n, PETSC_DECIDE, PETSC_DECIDE, nullptr, &G_));
  CHKERRTHROW(MatSetBlockSizes(G_, m_bs, n_bs));
  
  S_ = std::make_unique<PetscVector>(slip_block_size, num_local_elements, comm);
  t_boundary_ = std::make_unique<PetscVector>(m_bs, num_local_elements, comm);
  
  current_gf_ = 0;
  CHKERRTHROW(VecGetSize(S_->vec(), &n_gf_));
}

void SeasQDDiscreteGreenOperator::write_discrete_greens_operator() {
  PetscViewer v;
  PetscLogDouble t0,t1;
  
  CHKERRTHROW(PetscTime(&t0));
  CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_),gf_operator_filename_.c_str(),FILE_MODE_WRITE,&v));
  CHKERRTHROW(PetscViewerBinaryWrite(v,&current_gf_,1,PETSC_INT));
  CHKERRTHROW(MatView(G_,v));
  CHKERRTHROW(PetscViewerDestroy(&v));
  CHKERRTHROW(PetscTime(&t1));
  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"write_discrete_greens_operator() %1.2e (sec)\n",(double)(t1 - t0)));
  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"  status: computed %D / pending %D\n",current_gf_, n_gf_ - current_gf_));
}

void SeasQDDiscreteGreenOperator::load_discrete_greens_operator() {
  PetscViewer v;
  PetscBool found = PETSC_FALSE;
  PetscLogDouble t0,t1;

  if (!G_) {
    // error - todo
  }
  CHKERRTHROW(PetscTime(&t0));
  CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_),gf_operator_filename_.c_str(),FILE_MODE_READ,&v));
  CHKERRTHROW(PetscViewerBinaryRead(v,&current_gf_,1,NULL,PETSC_INT));
  CHKERRTHROW(MatLoad(G_,v));
  CHKERRTHROW(PetscViewerDestroy(&v));
  CHKERRTHROW(PetscTime(&t1));
  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"load_discrete_greens_operator() %1.2e (sec)\n",(double)(t1 - t0)));
  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"  status: loaded %D / pending %D\n",current_gf_, n_gf_ - current_gf_));
}

void SeasQDDiscreteGreenOperator::partial_assemble_discrete_greens_function() {
  auto slip_block_size = base::friction().slip_block_size();

  PetscInt num_local_elements = base::adapter().num_local_elements();
  PetscInt m_bs = base::adapter().traction_block_size();
  PetscInt n_bs = 1;
  PetscInt m = num_local_elements * m_bs;
  PetscInt n = num_local_elements * slip_block_size * n_bs;
  PetscInt mb_offset = 0;
  PetscInt nb_offset = 0;
  PetscInt start = current_gf_;
  PetscInt N = n_gf_;
  Stopwatch sw;
  double solve_time;
  int rank;
  MPI_Comm comm = base::comm();

  if (start == N) return;

  MPI_Comm_rank(comm, &rank);
  MPI_Scan(&num_local_elements, &mb_offset, 1, MPIU_INT, MPI_SUM, comm);
  mb_offset -= num_local_elements;
  MPI_Scan(&n, &nb_offset, 1, MPIU_INT, MPI_SUM, comm);
  nb_offset -= n;

  auto scatter = Scatter(base::adapter().fault_map().scatter_plan());
  auto ghost = scatter.template recv_prototype<double>(slip_block_size, ALIGNMENT);

  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"partial_assemble_discrete_greens_function() [%D , %D)\n",start,N));
  solve_time = 0.0;
  for (PetscInt i = start; i < N; ++i) {

    CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"Computing Green's function %D / %D\n",i,N));
    sw.start();
    CHKERRTHROW(VecZeroEntries(S_->vec()));
    if (i >= nb_offset && i < nb_offset + m) {
      PetscScalar one = 1.0;
      CHKERRTHROW(VecSetValue(S_->vec(), i, one, INSERT_VALUES));
    }
    S_->begin_assembly();
    S_->end_assembly();

    scatter.begin_scatter(*S_, ghost);
    scatter.wait_scatter();

    auto S_view = LocalGhostCompositeView(*S_, ghost);
    base::solve(0.0, S_view);
    base::update_traction(S_view);

    auto traction_handle = base::traction_.begin_access_readonly();
    for (std::size_t faultNo = 0; faultNo < num_local_elements; ++faultNo) {
      PetscInt g_m = mb_offset + faultNo;
      PetscInt g_n = i;
      auto traction_block = traction_handle.subtensor(slice{}, faultNo);
      CHKERRTHROW(
                  MatSetValuesBlocked(G_, 1, &g_m, 1, &g_n, traction_block.data(), INSERT_VALUES));
    }
    base::traction_.end_access_readonly(traction_handle);
    solve_time += sw.stop();

    current_gf_ = i + 1;

    /* checkpoint */
    MPI_Bcast(&solve_time,1,MPI_DOUBLE,0,comm);
    if (solve_time/60.0 > checkpoint_every_nmins_) {
      write_discrete_greens_operator();
      solve_time = 0.0;
    }
  }

  CHKERRTHROW(MatAssemblyBegin(G_, MAT_FINAL_ASSEMBLY));
  CHKERRTHROW(MatAssemblyEnd(G_, MAT_FINAL_ASSEMBLY));
}

void SeasQDDiscreteGreenOperator::get_discrete_greens_function() {
  PetscBool found = PETSC_FALSE;

  if (!G_) {
    create_discrete_greens_function();
  }

  CHKERRTHROW(PetscTestFile(gf_operator_filename_.c_str(),'r',&found));
  if (found) {
    load_discrete_greens_operator();
  }

  partial_assemble_discrete_greens_function();

  if (!found) {
    write_discrete_greens_operator();
  }
}

void SeasQDDiscreteGreenOperator::write_discrete_greens_traction() {
  PetscViewer v;
  PetscLogDouble t0,t1;

  CHKERRTHROW(PetscTime(&t0));
  CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_),gf_traction_filename_.c_str(),FILE_MODE_WRITE,&v));
  CHKERRTHROW(VecView(t_boundary_->vec(),v));
  CHKERRTHROW(PetscViewerDestroy(&v));
  CHKERRTHROW(PetscTime(&t1));
  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"write_discrete_greens_traction() %1.2e (sec)\n",(double)(t1 - t0)));
}

void SeasQDDiscreteGreenOperator::load_discrete_greens_traction() {
  PetscViewer v;
  PetscBool found = PETSC_FALSE;
  PetscLogDouble t0,t1;

  CHKERRTHROW(PetscTime(&t0));
  CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_),gf_traction_filename_.c_str(),FILE_MODE_READ,&v));
  CHKERRTHROW(VecLoad(base::traction_.vec(),v));
  CHKERRTHROW(PetscViewerDestroy(&v));
  CHKERRTHROW(PetscTime(&t1));
  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"load_discrete_greens_operator() %1.2e (sec)\n",(double)(t1 - t0)));
  CHKERRTHROW(VecCopy(base::traction_.vec(), t_boundary_->vec()));
}

void SeasQDDiscreteGreenOperator::get_boundary_traction() {
  PetscBool found = PETSC_FALSE;

  CHKERRTHROW(PetscTestFile(gf_traction_filename_.c_str(),'r',&found));
  if (found) {
    load_discrete_greens_traction();
  }

  if (!found) {
    compute_boundary_traction();

    write_discrete_greens_traction();
  }
}


} // namespace tndm
