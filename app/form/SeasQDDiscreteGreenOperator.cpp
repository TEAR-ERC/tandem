#include "SeasQDDiscreteGreenOperator.h"
#include "common/PetscUtil.h"

#include "form/RefElement.h"
#include "parallel/LocalGhostCompositeView.h"
#include "util/Stopwatch.h"

#include <cassert>
#include <filesystem>
namespace fs = std::filesystem;

namespace tndm {

SeasQDDiscreteGreenOperator::SeasQDDiscreteGreenOperator(
    std::unique_ptr<typename base::dg_t> dgop, std::unique_ptr<AbstractAdapterOperator> adapter,
    std::unique_ptr<AbstractFrictionOperator> friction,
    LocalSimplexMesh<DomainDimension> const& mesh, std::optional<std::string> prefix,
    double gf_checkpoint_every_nmins, bool matrix_free, MGConfig const& mg_config)
    : base(std::move(dgop), std::move(adapter), std::move(friction), matrix_free, mg_config) {

    int rank;

    MPI_Comm_rank(base::comm(), &rank);
    // if prefix is not empty, set filenames and mark checkpoint_enabled_ = true

    checkpoint_every_nmins_ = gf_checkpoint_every_nmins;

    if (prefix.has_value()) {
        std::string sprefix = prefix.value_or("");
        if (rank == 0) {
            std::cout << "Using GF checkpoint path: " << sprefix << std::endl;
        }
        auto pckp = fs::path(sprefix);
        if (rank == 0) {
            bool exists = fs::exists(pckp);
            if (!exists) {
                bool ret = fs::create_directories(pckp);
                if (!ret)
                    throw std::runtime_error("Failed to create directory!");
            }
        }
        auto prepend_checkpoint_path = [&](std::string pre_path) {
            fs::path pckpOp(pre_path);
            pckpOp /= gf_operator_filename_;
            gf_operator_filename_ = pckpOp;

            fs::path pckpVec(pre_path);
            pckpVec /= gf_traction_filename_;
            gf_traction_filename_ = pckpVec;

            fs::path pckpFacet(pre_path);
            pckpFacet /= gf_facet_filename_;
            gf_facet_filename_ = pckpFacet;
        };

        prepend_checkpoint_path(sprefix);
        checkpoint_enabled_ = true;
    }
    if (!checkpoint_enabled_) {
        compute_discrete_greens_function();
    } else {
        get_discrete_greens_function(mesh);
    }
}

SeasQDDiscreteGreenOperator::~SeasQDDiscreteGreenOperator() {
    MatDestroy(&G_);
    ISDestroy(&is_perm_);
}

void SeasQDDiscreteGreenOperator::set_boundary(
    std::unique_ptr<AbstractFacetFunctionalFactory> fun) {
    base::set_boundary(std::move(fun));
    if (!checkpoint_enabled_) {
        compute_boundary_traction();
    } else {
        get_boundary_traction();
    }
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

PetscInt SeasQDDiscreteGreenOperator::create_discrete_greens_function() {
    auto slip_block_size = base::friction().slip_block_size();
    PetscInt M, N, n_gf;
    PetscInt num_local_elements = base::adapter().num_local_elements();
    PetscInt m_bs = base::adapter().traction_block_size();
    PetscInt n_bs = 1;
    PetscInt m = num_local_elements * m_bs;
    PetscInt n = num_local_elements * slip_block_size * n_bs;
    PetscInt mb_offset = 0;
    PetscInt nb_offset = 0;
    int rank;
    MPI_Comm comm = base::comm();

    CHKERRTHROW(PetscPrintf(comm, "create_discrete_greens_function()\n"));
    MPI_Comm_rank(comm, &rank);
    MPI_Scan(&num_local_elements, &mb_offset, 1, MPIU_INT, MPI_SUM, comm);
    mb_offset -= num_local_elements;
    MPI_Scan(&n, &nb_offset, 1, MPIU_INT, MPI_SUM, comm);
    nb_offset -= n;

    CHKERRTHROW(MatCreateDense(comm, m, n, PETSC_DECIDE, PETSC_DECIDE, nullptr, &G_));
    CHKERRTHROW(MatSetBlockSizes(G_, m_bs, n_bs));
    CHKERRTHROW(MatGetSize(G_, &M, &N));
    CHKERRTHROW(PetscPrintf(comm, "Green's function operator size: %D x %D\n", M, N));

    S_ = std::make_unique<PetscVector>(slip_block_size, num_local_elements, comm);
    t_boundary_ = std::make_unique<PetscVector>(m_bs, num_local_elements, comm);

    CHKERRTHROW(VecGetSize(S_->vec(), &n_gf));
    return n_gf;
}

void SeasQDDiscreteGreenOperator::write_discrete_greens_operator(
    LocalSimplexMesh<DomainDimension> const& mesh, PetscInt current_gf, PetscInt n_gf) {
    PetscViewer v;
    PetscLogDouble t0, t1;
    int commsize;

    MPI_Comm_size(base::comm(), &commsize);

    CHKERRTHROW(PetscTime(&t0));
    CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_),
                                      gf_operator_filename_.c_str(), FILE_MODE_WRITE, &v));

    {
        PetscInt commsize_checkpoint = (PetscInt)commsize;
        CHKERRTHROW(PetscViewerBinaryWrite(v, &commsize_checkpoint, 1, PETSC_INT));
    }

    CHKERRTHROW(PetscViewerBinaryWrite(v, &current_gf, 1, PETSC_INT));

    CHKERRTHROW(MatView(G_, v));

    CHKERRTHROW(PetscViewerDestroy(&v));
    CHKERRTHROW(PetscTime(&t1));
    CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                            "write_discrete_greens_operator():matrix %1.2e (sec)\n",
                            (double)(t1 - t0)));
    CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                            "  status: computed %D / pending %D\n", current_gf, n_gf - current_gf));

    write_facet_labels_IS(mesh);
}

void SeasQDDiscreteGreenOperator ::write_facet_labels_IS(
    LocalSimplexMesh<DomainDimension> const& mesh) {
    PetscViewer v;
    PetscLogDouble t0, t1;
    std::size_t bndNo, d;
    auto const& fault_map = base::adapter().fault_map();
    PetscInt nfacets = (PetscInt)fault_map.local_size();
    constexpr PetscInt facet_size = (PetscInt)(DomainDimension);
    PetscInt* idx_;
    IS is;

    CHKERRTHROW(PetscCalloc1(nfacets * facet_size, &idx_));

    auto const& aFacets = mesh.facets();
    for (bndNo = 0; bndNo < nfacets; ++bndNo) {
        auto fctNo = fault_map.fctNo(bndNo);
        auto facets = aFacets[fctNo];
        static_assert(facet_size == facets.size());
        for (d = 0; d < facet_size; d++) {
            idx_[bndNo * facet_size + d] = (PetscInt)facets[d];
        }
    }

    CHKERRTHROW(ISCreateGeneral(PetscObjectComm((PetscObject)G_), nfacets * facet_size,
                                (const PetscInt*)idx_, PETSC_USE_POINTER, &is));

    CHKERRTHROW(PetscTime(&t0));
    CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_), gf_facet_filename_.c_str(),
                                      FILE_MODE_WRITE, &v));
    CHKERRTHROW(ISView(is, v));
    CHKERRTHROW(PetscViewerDestroy(&v));
    CHKERRTHROW(PetscTime(&t1));
    CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                            "write_discrete_greens_operator():facets %1.2e (sec)\n",
                            (double)(t1 - t0)));

    CHKERRTHROW(ISDestroy(&is));
    CHKERRTHROW(PetscFree(idx_));
}

IS SeasQDDiscreteGreenOperator ::load_facet_labels_seq_IS(void) {
    PetscViewer v;
    PetscLogDouble t0, t1;
    int rank;
    IS is = NULL;

    MPI_Comm_rank(base::comm(), &rank);
    if (rank == 0) {
        CHKERRTHROW(ISCreate(PETSC_COMM_SELF, &is));
        CHKERRTHROW(PetscTime(&t0));
        CHKERRTHROW(
            PetscViewerBinaryOpen(PETSC_COMM_SELF, gf_facet_filename_.c_str(), FILE_MODE_READ, &v));
        CHKERRTHROW(ISLoad(is, v));
        CHKERRTHROW(PetscViewerDestroy(&v));
        CHKERRTHROW(PetscTime(&t1));
        CHKERRTHROW(PetscPrintf(PETSC_COMM_SELF,
                                "load_discrete_greens_operator():facets %1.2e (sec)\n",
                                (double)(t1 - t0)));
    }
    return is;
}

void SeasQDDiscreteGreenOperator::create_permutation_redundant_IS(
    LocalSimplexMesh<DomainDimension> const& mesh, IS is) {
    MPI_Comm comm = base::comm();
    auto const& fault_map = base::adapter().fault_map();
    PetscInt nfacets = (PetscInt)fault_map.local_size();
    constexpr PetscInt facet_size = (PetscInt)(DomainDimension);
    PetscInt *idx_ = NULL, is_len;
    PetscInt* fault_map_index = NULL;
    int rank;

    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        CHKERRTHROW(ISGetSize(is, &is_len));
        CHKERRTHROW(ISGetIndices(is, (const PetscInt**)&idx_));
    }

    MPI_Bcast((void*)&is_len, 1, MPIU_INT, 0, comm);

    if (!idx_) {
        CHKERRTHROW(PetscCalloc1(is_len, &idx_));
    }

    MPI_Bcast((void*)idx_, is_len, MPIU_INT, 0, comm);

    CHKERRTHROW(PetscCalloc1(nfacets, &fault_map_index));

    using map_t = std::unordered_map<Simplex<DomainDimension - 1u>, std::size_t,
                                     SimplexHash<DomainDimension - 1u>>;
    auto map = map_t(nfacets);
    std::size_t map_index = 0;
    for (PetscInt n = 0; n < is_len / facet_size; n++) {
        auto plex = Simplex<DomainDimension - 1u>{};
        for (PetscInt d = 0; d < facet_size; d++) {
            plex[d] = idx_[n * facet_size + d];
        }
        map[plex] = map_index++;
    }

    for (std::size_t bndNo = 0; bndNo < nfacets; ++bndNo) {
        auto fctNo = fault_map.fctNo(bndNo);
        auto it = map.find(mesh.facets()[fctNo]);
        if (it == map.end()) {
            throw std::runtime_error("facet not found when compiling permutation index set");
        }
        fault_map_index[bndNo] = it->second;
    }

    if (rank != 0) {
        CHKERRTHROW(PetscFree(idx_));
    } else {
        CHKERRTHROW(ISRestoreIndices(is, (const PetscInt**)&idx_));
    }

    {
        CHKERRTHROW(ISCreateGeneral(comm, nfacets, (const PetscInt*)fault_map_index,
                                    PETSC_COPY_VALUES, &is_perm_));
        CHKERRTHROW(PetscFree(fault_map_index));
    }
}

std::tuple<Mat, Mat>
SeasQDDiscreteGreenOperator ::create_row_col_permutation_matrices(bool create_row,
                                                                  bool create_col) {
    MPI_Comm comm = base::comm();
    PetscInt num_local_elements = base::adapter().num_local_elements();
    PetscInt num_global_elements;
    PetscInt nfacets = (PetscInt)base::adapter().fault_map().local_size();
    PetscInt M, N, m, n, br, bc, d;
    Mat Rperm = NULL, Cperm = NULL;
    PetscInt mb_offset = 0;
    const PetscInt* fault_map_index;

    CHKERRTHROW(ISGetIndices(is_perm_, &fault_map_index));

    MPI_Scan(&num_local_elements, &mb_offset, 1, MPIU_INT, MPI_SUM, comm);
    mb_offset -= num_local_elements;

    MPI_Allreduce(&num_local_elements, &num_global_elements, 1, MPIU_INT, MPI_SUM, comm);

    CHKERRTHROW(MatGetSize(G_, &M, &N));
    CHKERRTHROW(MatGetLocalSize(G_, &m, &n));

    br = M / num_global_elements;
    bc = N / num_global_elements;

    assert(M % num_global_elements == 0);
    assert(N % num_global_elements == 0);

    // G = Rperm G_ Cperm
    // Rperm is the transpose of Cperm just with a different block size

    auto create_permutation_matrix = [this, &comm, mb_offset, &fault_map_index](
                                         PetscInt m, PetscInt M, PetscInt block_size, bool swap) {
        Mat perm = NULL;
        CHKERRTHROW(MatCreate(comm, &perm));
        CHKERRTHROW(MatSetSizes(perm, m, m, M, M));
        CHKERRTHROW(MatSetType(perm, MATAIJ));
        CHKERRTHROW(MatSeqAIJSetPreallocation(perm, block_size, NULL));
        CHKERRTHROW(MatMPIAIJSetPreallocation(perm, block_size, NULL, block_size, NULL));

        for (std::size_t bndNo = 0; bndNo < this->adapter().fault_map().local_size(); ++bndNo) {
            PetscInt from = mb_offset + bndNo;
            PetscInt to = fault_map_index[bndNo];
            if (swap) {
                std::swap(from, to);
            }
            for (PetscInt d = 0; d < block_size; d++) {
                CHKERRTHROW(MatSetValue(perm, block_size * from + d, block_size * to + d, 1.0,
                                        INSERT_VALUES));
            }
        }
        CHKERRTHROW(MatAssemblyBegin(perm, MAT_FINAL_ASSEMBLY));
        CHKERRTHROW(MatAssemblyEnd(perm, MAT_FINAL_ASSEMBLY));
        return perm;
    };

    if (create_row) {
        Rperm = create_permutation_matrix(m, M, br, false);
    }

    if (create_col) {
        Cperm = create_permutation_matrix(n, N, bc, true);
    }

    return std::make_tuple(Rperm, Cperm);
}

PetscInt SeasQDDiscreteGreenOperator::load_discrete_greens_operator(
    LocalSimplexMesh<DomainDimension> const& mesh, PetscInt n_gf) {
    PetscViewer v;
    PetscLogDouble t0, t1;
    MPI_Comm comm = base::comm();
    int commsize;
    PetscInt commsize_checkpoint;
    PetscInt current_gf = 0;

    if (!G_) {
        CHKERRTHROW(PetscPrintf(base::comm(), "G_ is NULL. Cannot load the operator. Must call "
                                              "create_discrete_greens_function() first!\n"));
        throw std::runtime_error("G_ is NULL");
    }
    CHKERRTHROW(PetscTime(&t0));
    CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_),
                                      gf_operator_filename_.c_str(), FILE_MODE_READ, &v));

    CHKERRTHROW(PetscViewerBinaryRead(v, &commsize_checkpoint, 1, NULL, PETSC_INT));
    MPI_Comm_size(comm, &commsize);
    if ((PetscInt)commsize != commsize_checkpoint) {
        CHKERRTHROW(PetscPrintf(comm,
                                "GF loaded was created with commsize = %d. Current commsize = %d. "
                                "Repartitioning required.\n",
                                (int)commsize_checkpoint, (int)commsize));
        repartition_gfs_ = true;
    } else {
        CHKERRTHROW(PetscPrintf(
            comm, "GF loaded was created with commsize matching current (%d).\n", (int)commsize));
        repartition_gfs_ = false;
    }

    CHKERRTHROW(PetscViewerBinaryRead(v, &current_gf, 1, NULL, PETSC_INT));

    CHKERRTHROW(MatLoad(G_, v));
    CHKERRTHROW(PetscViewerDestroy(&v));
    CHKERRTHROW(PetscTime(&t1));
    CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                            "load_discrete_greens_operator() %1.2e (sec)\n", (double)(t1 - t0)));
    CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_), "  status: loaded %D / pending %D\n",
                            current_gf, n_gf - current_gf));

    if (repartition_gfs_) {
        IS is = load_facet_labels_seq_IS();
        create_permutation_redundant_IS(mesh, is);
        CHKERRTHROW(ISDestroy(&is));

        Mat GCperm;
        auto [Rperm, Cperm] = create_row_col_permutation_matrices(PETSC_TRUE, PETSC_TRUE);

        CHKERRTHROW(MatMatMult(G_, Cperm, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GCperm));
        CHKERRTHROW(MatMatMult(Rperm, GCperm, MAT_REUSE_MATRIX, PETSC_DEFAULT, &G_));
        CHKERRTHROW(MatDestroy(&GCperm));
        CHKERRTHROW(MatDestroy(&Rperm));
        CHKERRTHROW(MatDestroy(&Cperm));
    }
    return current_gf;
}

void SeasQDDiscreteGreenOperator::partial_assemble_discrete_greens_function(
    LocalSimplexMesh<DomainDimension> const& mesh, PetscInt current_gf, PetscInt n_gf) {
    auto slip_block_size = base::friction().slip_block_size();

    PetscInt num_local_elements = base::adapter().num_local_elements();
    PetscInt m_bs = base::adapter().traction_block_size();
    PetscInt n_bs = 1;
    PetscInt m = num_local_elements * m_bs;
    PetscInt n = num_local_elements * slip_block_size * n_bs;
    PetscInt mb_offset = 0;
    PetscInt nb_offset = 0;
    PetscInt start = current_gf;
    PetscInt N = n_gf;
    Stopwatch sw;
    double solve_time;
    int rank;
    MPI_Comm comm = base::comm();

    if (start == N)
        return;

    MPI_Comm_rank(comm, &rank);
    MPI_Scan(&num_local_elements, &mb_offset, 1, MPIU_INT, MPI_SUM, comm);
    mb_offset -= num_local_elements;
    MPI_Scan(&n, &nb_offset, 1, MPIU_INT, MPI_SUM, comm);
    nb_offset -= n;

    auto scatter = Scatter(base::adapter().fault_map().scatter_plan());
    auto ghost = scatter.template recv_prototype<double>(slip_block_size, ALIGNMENT);

    CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                            "partial_assemble_discrete_greens_function() [%D , %D)\n", start, N));
    solve_time = 0.0;
    for (PetscInt i = start; i < N; ++i) {

        CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                                "Computing Green's function %D / %D\n", i, N));
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

        current_gf = i + 1;

        /* checkpoint */
        MPI_Bcast(&solve_time, 1, MPI_DOUBLE, 0, comm);
        if (solve_time / 60.0 > checkpoint_every_nmins_) {
            CHKERRTHROW(MatAssemblyBegin(G_, MAT_FINAL_ASSEMBLY));
            CHKERRTHROW(MatAssemblyEnd(G_, MAT_FINAL_ASSEMBLY));
            write_discrete_greens_operator(mesh, current_gf, n_gf);
            solve_time = 0.0;
        }
    }

    CHKERRTHROW(MatAssemblyBegin(G_, MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd(G_, MAT_FINAL_ASSEMBLY));
}

void SeasQDDiscreteGreenOperator::get_discrete_greens_function(
    LocalSimplexMesh<DomainDimension> const& mesh) {
    PetscBool found = PETSC_FALSE;
    PetscInt n_gfloaded = 0, n_gf;

    // Create an empty dense matrix to store all GFs
    if (!G_) {
        n_gf = create_discrete_greens_function();
    }

    // If a checkpoint file is found, load it. Record the number of assembled GFs found in file.
    CHKERRTHROW(PetscTestFile(gf_operator_filename_.c_str(), 'r', &found));
    if (found) {
        n_gfloaded = load_discrete_greens_operator(mesh, n_gf);
    }

    // Assemble as many GFs as possible in the range [current_gf, n_gf)
    partial_assemble_discrete_greens_function(mesh, n_gfloaded, n_gf);

    // Write out the operator whenever the fully assembled operator was not loaded from file
    if (n_gfloaded != n_gf) {
        write_discrete_greens_operator(mesh, n_gf, n_gf);
    }
}

void SeasQDDiscreteGreenOperator::write_discrete_greens_traction() {
    PetscViewer v;
    PetscLogDouble t0, t1;

    CHKERRTHROW(PetscTime(&t0));
    CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_),
                                      gf_traction_filename_.c_str(), FILE_MODE_WRITE, &v));
    CHKERRTHROW(VecView(t_boundary_->vec(), v));
    CHKERRTHROW(PetscViewerDestroy(&v));
    CHKERRTHROW(PetscTime(&t1));
    CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                            "write_discrete_greens_traction() %1.2e (sec)\n", (double)(t1 - t0)));
}

void SeasQDDiscreteGreenOperator::load_discrete_greens_traction() {
    PetscViewer v;
    PetscLogDouble t0, t1;

    CHKERRTHROW(PetscTime(&t0));
    CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_),
                                      gf_traction_filename_.c_str(), FILE_MODE_READ, &v));
    CHKERRTHROW(VecLoad(base::traction_.vec(), v));
    CHKERRTHROW(PetscViewerDestroy(&v));
    CHKERRTHROW(PetscTime(&t1));
    CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                            "load_discrete_greens_operator() %1.2e (sec)\n", (double)(t1 - t0)));

    if (repartition_gfs_) {
        Vec tmp;

        CHKERRTHROW(VecDuplicate(base::traction_.vec(), &tmp));
        auto [Rperm, Cperm] = create_row_col_permutation_matrices(true, false);
        CHKERRTHROW(MatMult(Rperm, base::traction_.vec(), tmp));
        CHKERRTHROW(MatDestroy(&Rperm));
        CHKERRTHROW(VecCopy(tmp, base::traction_.vec()));
        CHKERRTHROW(VecDestroy(&tmp));
    }
    CHKERRTHROW(VecCopy(base::traction_.vec(), t_boundary_->vec()));
}

void SeasQDDiscreteGreenOperator::get_boundary_traction() {
    PetscBool found = PETSC_FALSE;

    CHKERRTHROW(PetscTestFile(gf_traction_filename_.c_str(), 'r', &found));
    if (found) {
        load_discrete_greens_traction();
    }

    if (!found) {
        compute_boundary_traction();

        write_discrete_greens_traction();
    }
}

} // namespace tndm
