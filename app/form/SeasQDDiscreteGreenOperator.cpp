#include "SeasQDDiscreteGreenOperator.h"
#include "common/PetscUtil.h"

#include "form/RefElement.h"
#include "parallel/LocalGhostCompositeView.h"
#include "util/Stopwatch.h"

#include <filesystem>
namespace fs = std::filesystem;

namespace tndm {

SeasQDDiscreteGreenOperator::SeasQDDiscreteGreenOperator(
    std::unique_ptr<typename base::dg_t> dgop, std::unique_ptr<AbstractAdapterOperator> adapter,
    std::unique_ptr<AbstractFrictionOperator> friction,
    LocalSimplexMesh<DomainDimension> const& mesh,
    bool matrix_free, MGConfig const& mg_config, std::string prefix)
    : base(std::move(dgop), std::move(adapter), std::move(friction), matrix_free, mg_config) {

    int rank;

    MPI_Comm_rank(base::comm(),&rank);
    // if prefix is not empty, set filenames and mark checkpoint_enabled_ = true
    if (!prefix.empty()) {
        if (rank == 0) { std::cout << "Using GF checkpoint path: " << prefix << std::endl; }
        fs::path pckp(prefix);
        if (rank == 0) {
          bool exists = fs::exists(pckp);
          if (!exists) {
              bool ret = fs::create_directories(pckp);
              if (!ret) std::cout << "--> Failed to create directory!" << std::endl;
          }
        }
        prepend_checkpoint_path(prefix);
        checkpoint_enabled_ = true;
    }
    if (!checkpoint_enabled_) {
        compute_discrete_greens_function();
    } else {
        get_discrete_greens_function(mesh);
    }
}

SeasQDDiscreteGreenOperator::~SeasQDDiscreteGreenOperator() { MatDestroy(&G_); ISDestroy(&is_perm_); }

void SeasQDDiscreteGreenOperator::set_boundary(
    std::unique_ptr<AbstractFacetFunctionalFactory> fun) {
    base::set_boundary(std::move(fun));
    if (!checkpoint_enabled_) {
        compute_boundary_traction();
    } else {
        get_boundary_traction();
    }
}

  std::tuple<std::string, std::string> SeasQDDiscreteGreenOperator::get_checkpoint_filenames(void) {
    return std::make_tuple(gf_operator_filename_, gf_traction_filename_);
  }

  double SeasQDDiscreteGreenOperator::get_checkpoint_time_interval(void) {
    return checkpoint_every_nmins_;
  }

  void SeasQDDiscreteGreenOperator::set_checkpoint_filenames(std::string mat_fname, std::string vec_fname) {
    gf_operator_filename_ = mat_fname;
    gf_traction_filename_ = vec_fname;
  }

  void SeasQDDiscreteGreenOperator::prepend_checkpoint_path(std::string pre_path) {
    fs::path pckpOp(pre_path);
    pckpOp /= gf_operator_filename_;
    gf_operator_filename_ = pckpOp;

    fs::path pckpVec(pre_path);
    pckpVec /= gf_traction_filename_;
    gf_traction_filename_ = pckpVec;

    fs::path pckpFacet(pre_path);
    pckpFacet /= gf_facet_filename_;
    gf_facet_filename_ = pckpFacet;
  }

  void SeasQDDiscreteGreenOperator::set_checkpoint_time_interval(double t) {
    checkpoint_every_nmins_ = t;
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
  PetscInt M,N;
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
  CHKERRTHROW(MatGetSize(G_,&M,&N));
  CHKERRTHROW(PetscPrintf(comm,"Green's function operator size: %D x %D\n",M,N));

  S_ = std::make_unique<PetscVector>(slip_block_size, num_local_elements, comm);
  t_boundary_ = std::make_unique<PetscVector>(m_bs, num_local_elements, comm);

  current_gf_ = 0;
  CHKERRTHROW(VecGetSize(S_->vec(), &n_gf_));
}

void SeasQDDiscreteGreenOperator::write_discrete_greens_operator(LocalSimplexMesh<DomainDimension> const& mesh) {
  PetscViewer v;
  PetscLogDouble t0,t1;
  int commsize;

  MPI_Comm_size(base::comm(),&commsize);

  CHKERRTHROW(PetscTime(&t0));
  CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_),gf_operator_filename_.c_str(),FILE_MODE_WRITE,&v));

  {
    PetscInt _commsize = (PetscInt)commsize;
    CHKERRTHROW(PetscViewerBinaryWrite(v,&_commsize,1,PETSC_INT));
  }

  CHKERRTHROW(PetscViewerBinaryWrite(v,&current_gf_,1,PETSC_INT));

  CHKERRTHROW(MatView(G_,v));

  CHKERRTHROW(PetscViewerDestroy(&v));
  CHKERRTHROW(PetscTime(&t1));
  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"write_discrete_greens_operator():matrix %1.2e (sec)\n",(double)(t1 - t0)));
  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"  status: computed %D / pending %D\n",current_gf_, n_gf_ - current_gf_));

  write_facet_labels_IS(mesh);
}

void SeasQDDiscreteGreenOperator :: write_facet_labels_IS(LocalSimplexMesh<DomainDimension> const& mesh)
{
  PetscViewer v;
  PetscLogDouble t0,t1;
  std::size_t bndNo,d;
  PetscInt nfacets = (PetscInt)base::adapter().fault_map().local_size();
  PetscInt facet_size = (PetscInt)(DomainDimension);
  PetscInt *_idx;
  IS is;

  CHKERRTHROW(PetscCalloc1(nfacets*facet_size,&_idx));

  for (bndNo = 0; bndNo < base::adapter().fault_map().local_size(); ++bndNo) {
    auto fctNo = base::adapter().fault_map().fctNo(bndNo);
    auto facets = mesh.facets()[fctNo];
    for (d=0; d<facet_size; d++) {
      _idx[bndNo * facet_size + d] = (PetscInt)facets[d];
    }
  }

  CHKERRTHROW(ISCreateGeneral(PetscObjectComm((PetscObject)G_),nfacets*facet_size,(const PetscInt*)_idx,PETSC_USE_POINTER,&is));

  CHKERRTHROW(PetscTime(&t0));
  CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_),gf_facet_filename_.c_str(),FILE_MODE_WRITE,&v));
  CHKERRTHROW(PetscViewerDestroy(&v));
  CHKERRTHROW(PetscTime(&t1));
  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"write_discrete_greens_operator():facets %1.2e (sec)\n",(double)(t1 - t0)));

  CHKERRTHROW(ISDestroy(&is));
  CHKERRTHROW(PetscFree(_idx));
}

IS SeasQDDiscreteGreenOperator :: load_facet_labels_seq_IS(void)
{
  PetscViewer v;
  PetscLogDouble t0,t1;
  int rank;
  IS is = NULL;

  MPI_Comm_rank(base::comm(),&rank);
  if (rank == 0) {
    CHKERRTHROW(ISCreate(PETSC_COMM_SELF,&is));
    CHKERRTHROW(PetscTime(&t0));
    CHKERRTHROW(PetscViewerBinaryOpen(PETSC_COMM_SELF,gf_facet_filename_.c_str(),FILE_MODE_READ,&v));
    CHKERRTHROW(ISLoad(is,v));
    CHKERRTHROW(PetscViewerDestroy(&v));
    CHKERRTHROW(PetscTime(&t1));
    CHKERRTHROW(PetscPrintf(PETSC_COMM_SELF,"load_discrete_greens_operator():facets %1.2e (sec)\n",(double)(t1 - t0)));
  }
  return is;
}

void SeasQDDiscreteGreenOperator :: create_permutation_redundant_IS(LocalSimplexMesh<DomainDimension> const& mesh, IS is)
{
  MPI_Comm comm = base::comm();
  std::size_t bndNo;
  PetscInt nfacets = (PetscInt)base::adapter().fault_map().local_size();
  PetscInt facet_size = (PetscInt)(DomainDimension);
  PetscInt *_idx = NULL,is_len;
  PetscInt *fault_map_index = NULL;
  int rank;

  MPI_Comm_rank(comm,&rank);
  if (rank == 0) {
    CHKERRTHROW(ISGetSize(is,&is_len));
    CHKERRTHROW(ISGetIndices(is,(const PetscInt**)&_idx));
  }

  MPI_Bcast((void*)&is_len,1,MPIU_INT,0,comm);

  if (!_idx) {
    CHKERRTHROW(PetscCalloc1(is_len,&_idx));
  }

  MPI_Bcast((void*)_idx,is_len,MPIU_INT,0,comm);

  CHKERRTHROW(PetscCalloc1((PetscInt)base::adapter().fault_map().local_size(),&fault_map_index));

  for (bndNo = 0; bndNo < base::adapter().fault_map().local_size(); ++bndNo) {
    auto fctNo = base::adapter().fault_map().fctNo(bndNo);
    auto facets = mesh.facets()[fctNo];
    PetscInt facet_size = (PetscInt)(DomainDimension);
    PetscInt n,map_index = -1;

    for (n=0; n<is_len/facet_size; n++) {
      int found = 0;
      for (PetscInt d=0; d<facet_size; d++) {
        if ((PetscInt)facets[d] == _idx[n*facet_size+d]) {
          found++;
        }
      }
      if (found == facet_size) {map_index = n; break;}
    }
    fault_map_index[bndNo] = map_index;
    if (map_index == -1) {
      // error
    }
  }

  if (rank != 0) {
    CHKERRTHROW(PetscFree(_idx));
  } else {
    CHKERRTHROW(ISRestoreIndices(is,(const PetscInt**)&_idx));
  }

  {
    PetscInt len = (PetscInt)base::adapter().fault_map().local_size();
    CHKERRTHROW(ISCreateGeneral(comm,len,(const PetscInt*)fault_map_index,PETSC_COPY_VALUES,&is_perm_));
    CHKERRTHROW(PetscFree(fault_map_index));
  }
}

std::tuple<Mat, Mat> SeasQDDiscreteGreenOperator :: create_row_col_permutation_matrices(bool create_row, bool create_col)
{
  MPI_Comm comm = base::comm();
  PetscInt num_local_elements = base::adapter().num_local_elements();
  PetscInt num_global_elements;
  std::size_t bndNo;
  PetscInt nfacets = (PetscInt)base::adapter().fault_map().local_size();
  PetscInt M,N,m,n,br,bc,d;
  Mat Rperm = NULL,Cperm = NULL;
  PetscInt mb_offset = 0;
  const PetscInt *fault_map_index;

  CHKERRTHROW(ISGetIndices(is_perm_,&fault_map_index));

  MPI_Scan(&num_local_elements, &mb_offset, 1, MPIU_INT, MPI_SUM, comm);
  mb_offset -= num_local_elements;

  num_global_elements = num_local_elements;
  MPI_Allreduce(MPI_IN_PLACE,&num_global_elements,1,MPIU_INT,MPI_SUM,comm);

  CHKERRTHROW(MatGetSize(G_,&M,&N));
  CHKERRTHROW(MatGetLocalSize(G_,&m,&n));

  br = M/num_global_elements;
  bc = N/num_global_elements;

  // G = Rperm G_ Cperm
  // Rperm is the transpose of Cperm just with a different block size
  if (create_row) {
    //CHKERRTHROW(MatCreateDense(comm, m, m, M, M, nullptr, &Rperm));
    CHKERRTHROW(MatCreate(comm,&Rperm));
    CHKERRTHROW(MatSetSizes(Rperm,m,m,M,M));
    CHKERRTHROW(MatSetType(Rperm,MATAIJ));
    CHKERRTHROW(MatSeqAIJSetPreallocation(Rperm,br,NULL));
    CHKERRTHROW(MatMPIAIJSetPreallocation(Rperm,br,NULL,br,NULL));

    for (bndNo = 0; bndNo < base::adapter().fault_map().local_size(); ++bndNo) {
      PetscInt from = mb_offset + bndNo;
      PetscInt to = fault_map_index[bndNo];
      for (d=0; d<br; d++) {
        CHKERRTHROW(MatSetValue(Rperm,br * from + d,br * to + d,1.0,INSERT_VALUES));
      }
    }
    CHKERRTHROW(MatAssemblyBegin(Rperm,MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd(Rperm,MAT_FINAL_ASSEMBLY));
  }

  if (create_col) {
    CHKERRTHROW(MatCreate(comm,&Cperm));
    CHKERRTHROW(MatSetSizes(Cperm,n,n,N,N));
    CHKERRTHROW(MatSetType(Cperm,MATAIJ));
    CHKERRTHROW(MatSeqAIJSetPreallocation(Cperm,bc,NULL));
    CHKERRTHROW(MatMPIAIJSetPreallocation(Cperm,bc,NULL,bc,NULL));

    for (bndNo = 0; bndNo < base::adapter().fault_map().local_size(); ++bndNo) {
      PetscInt from = mb_offset + bndNo;
      PetscInt to = fault_map_index[bndNo];
      for (d=0; d<bc; d++) {
        CHKERRTHROW(MatSetValue(Cperm,bc * to + d,bc * from + d,1.0,INSERT_VALUES));
      }
    }
    CHKERRTHROW(MatAssemblyBegin(Cperm,MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd(Cperm,MAT_FINAL_ASSEMBLY));
  }

  CHKERRTHROW(ISRestoreIndices(is_perm_,&fault_map_index));

  return std::make_tuple(Rperm, Cperm);
}

void SeasQDDiscreteGreenOperator :: load_discrete_greens_operator(LocalSimplexMesh<DomainDimension> const& mesh) {
  PetscViewer v;
  PetscLogDouble t0,t1;
  MPI_Comm comm = base::comm();
  int commsize;
  PetscInt _commsize;

  if (!G_) {
    CHKERRTHROW(PetscPrintf(base::comm(),"G_ is NULL. Cannot load the operator. Must call create_discrete_greens_function() first!\n"));
    throw std::runtime_error("G_ is NULL");
  }
  CHKERRTHROW(PetscTime(&t0));
  CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_),gf_operator_filename_.c_str(),FILE_MODE_READ,&v));

  CHKERRTHROW(PetscViewerBinaryRead(v,&_commsize,1,NULL,PETSC_INT));
  MPI_Comm_size(comm,&commsize);
  if ((PetscInt)commsize != _commsize) {
    CHKERRTHROW(PetscPrintf(comm,"GF loaded was created with commsize = %d. Current commsize = %d. Repartitioning required.\n",(int)_commsize,(int)commsize));
    repartition_gfs_ = true;
  } else {
    CHKERRTHROW(PetscPrintf(comm,"GF loaded was created with commsize matching current (%d).\n",(int)commsize));
    repartition_gfs_ = false;
  }

  CHKERRTHROW(PetscViewerBinaryRead(v,&current_gf_,1,NULL,PETSC_INT));

  CHKERRTHROW(MatLoad(G_,v));
  CHKERRTHROW(PetscViewerDestroy(&v));
  CHKERRTHROW(PetscTime(&t1));
  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"load_discrete_greens_operator() %1.2e (sec)\n",(double)(t1 - t0)));
  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"  status: loaded %D / pending %D\n",current_gf_, n_gf_ - current_gf_));

  if (repartition_gfs_) {
    IS is = load_facet_labels_seq_IS();
    create_permutation_redundant_IS(mesh, is);
    CHKERRTHROW(ISDestroy(&is));

    Mat GCperm;
    auto perm_mat = create_row_col_permutation_matrices(PETSC_TRUE, PETSC_TRUE);

    CHKERRTHROW(MatMatMult(G_,std::get<1>(perm_mat) ,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&GCperm));
    CHKERRTHROW(MatMatMult(std::get<0>(perm_mat),GCperm,MAT_REUSE_MATRIX,PETSC_DEFAULT,&G_));
    CHKERRTHROW(MatDestroy(&GCperm));
    CHKERRTHROW(MatDestroy(&std::get<0>(perm_mat)));
    CHKERRTHROW(MatDestroy(&std::get<1>(perm_mat)));
  }
}

void SeasQDDiscreteGreenOperator::partial_assemble_discrete_greens_function(LocalSimplexMesh<DomainDimension> const& mesh) {
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
      write_discrete_greens_operator(mesh);
      solve_time = 0.0;
    }
  }

  CHKERRTHROW(MatAssemblyBegin(G_, MAT_FINAL_ASSEMBLY));
  CHKERRTHROW(MatAssemblyEnd(G_, MAT_FINAL_ASSEMBLY));
}

void SeasQDDiscreteGreenOperator::get_discrete_greens_function(LocalSimplexMesh<DomainDimension> const& mesh) {
  PetscBool found = PETSC_FALSE;
  PetscInt n_gf_loaded = 0;

  // Create an empty dense matrix to store all GFs
  if (!G_) {
    create_discrete_greens_function();
  }

  // If a checkpoint file is found, load it. Record the number of assembled GFs found in file.
  CHKERRTHROW(PetscTestFile(gf_operator_filename_.c_str(),'r',&found));
  if (found) {
    load_discrete_greens_operator(mesh);
    n_gf_loaded = current_gf_;
  }

  // Assemble as many GFs as possible in the range [current_gf_, n_gf_)
  partial_assemble_discrete_greens_function(mesh);

  // Write out the operator whenever the fully assembled operator was not loaded from file
  if (n_gf_loaded != n_gf_) {
    write_discrete_greens_operator(mesh);
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
  PetscLogDouble t0,t1;

  CHKERRTHROW(PetscTime(&t0));
  CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_),gf_traction_filename_.c_str(),FILE_MODE_READ,&v));
  CHKERRTHROW(VecLoad(base::traction_.vec(),v));
  CHKERRTHROW(PetscViewerDestroy(&v));
  CHKERRTHROW(PetscTime(&t1));
  CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),"load_discrete_greens_operator() %1.2e (sec)\n",(double)(t1 - t0)));

  if (repartition_gfs_) {
    Vec tmp;

    CHKERRTHROW(VecDuplicate(base::traction_.vec(),&tmp));
    auto perm_mat = create_row_col_permutation_matrices(true, false);
    CHKERRTHROW(MatMult(std::get<0>(perm_mat),base::traction_.vec(),tmp));
    CHKERRTHROW(MatDestroy(&std::get<0>(perm_mat)));
    CHKERRTHROW(VecCopy(tmp,base::traction_.vec()));
    CHKERRTHROW(VecDestroy(&tmp));
  }
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
