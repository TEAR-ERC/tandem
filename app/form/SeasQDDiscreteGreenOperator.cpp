#include "SeasQDDiscreteGreenOperator.h"
#include "common/PetscLoggingUtils.h"
#include "common/PetscUtil.h"

#include "form/RefElement.h"
#include "parallel/LocalGhostCompositeView.h"
#include "util/Stopwatch.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace fs = std::filesystem;

namespace tndm {

GfSolver::GfSolver(std::unique_ptr<dg_t> dgop, std::unique_ptr<AbstractAdapterOperator> adapter,
                   std::unique_ptr<AbstractFrictionOperator> friction, bool matrix_free,
                   MGConfig const& mg_config)
    : SeasQDOperator(std::move(dgop), std::move(adapter), std::move(friction), matrix_free,
                     mg_config),
      scatter_(SeasQDOperator::adapter().fault_map().scatter_plan()),
      ghost_(
          scatter_.recv_prototype<double>(SeasQDOperator::friction().slip_block_size(), ALIGNMENT)),
      slip_(SeasQDOperator::friction().slip_block_size(),
            SeasQDOperator::adapter().num_local_elements(), SeasQDOperator::comm()) {}

void GfSolver::unit_slip_response(PetscInt global_slip_index) {
    CHKERRTHROW(VecZeroEntries(slip_.vec()));
    PetscInt first, last;
    CHKERRTHROW(VecGetOwnershipRange(slip_.vec(), &first, &last));
    if (global_slip_index >= first && global_slip_index < last) {
        PetscScalar one = 1.0;
        CHKERRTHROW(VecSetValue(slip_.vec(), global_slip_index, one, INSERT_VALUES));
    }
    slip_.begin_assembly();
    slip_.end_assembly();

    scatter_.begin_scatter(slip_, ghost_);
    scatter_.wait_scatter();

    auto slip_view = LocalGhostCompositeView(slip_, ghost_);
    SeasQDOperator::solve(0.0, slip_view);
    SeasQDOperator::update_traction(slip_view);
}

GreensFunctionIndices::GreensFunctionIndices(SeasQDDiscreteGreenOperator const& op) : n_bs(1) {
    slip_block_size = op.base::friction().slip_block_size();
    num_local_elements = op.base::adapter().num_local_elements();
    m_bs = op.base::adapter().traction_block_size();
    m = num_local_elements * m_bs;
    n = num_local_elements * slip_block_size * n_bs;
    comm = op.base::comm();
    MPI_Comm_rank(comm, &rank);

    mb_offset = 0;
    nb_offset = 0;
    MPI_Scan(&num_local_elements, &mb_offset, 1, MPIU_INT, MPI_SUM, comm);
    mb_offset -= num_local_elements;
    MPI_Scan(&n, &nb_offset, 1, MPIU_INT, MPI_SUM, comm);
    nb_offset -= n;
}

SeasQDDiscreteGreenOperator::SeasQDDiscreteGreenOperator(
    std::unique_ptr<typename base::dg_t> dgop, std::unique_ptr<AbstractAdapterOperator> adapter,
    std::unique_ptr<AbstractFrictionOperator> friction,
    LocalSimplexMesh<DomainDimension> const& mesh, std::optional<std::string> prefix,
    double gf_checkpoint_every_nmins, bool matrix_free, MGConfig const& mg_config,
    GfWorkerFactory worker_factory, unsigned gf_groups, bool balance_rows, bool single_precision)
    : base(std::move(dgop), std::move(adapter), std::move(friction), matrix_free, mg_config),
      worker_factory_(std::move(worker_factory)), gf_groups_(gf_groups),
      balance_rows_(balance_rows), single_precision_(single_precision) {

    int rank;

    CHKERRTHROW(PetscLogEventRegister("GFGatherSlip", MAT_CLASSID, &ev_gather_));
    CHKERRTHROW(PetscLogEventRegister("GFMultLocal", MAT_CLASSID, &ev_mult_));
    CHKERRTHROW(PetscLogEventRegister("GFRedistribute", MAT_CLASSID, &ev_redistribute_));
    CHKERRTHROW(PetscLogEventRegister("GFFrictionRhs", MAT_CLASSID, &ev_friction_));

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
    get_discrete_greens_function(mesh);
    if (single_precision_) {
        convert_to_single_precision();
    }
    // Only needed for the computation above (and refers to objects that may not live on)
    worker_factory_ = nullptr;
}

SeasQDDiscreteGreenOperator::~SeasQDDiscreteGreenOperator() {
    VecScatterDestroy(&gather_slip_);
    VecDestroy(&slip_all_);
    VecScatterDestroy(&to_fault_layout_);
    VecDestroy(&y_balanced_);
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

    if (single_precision_) {
        matmult_single_precision(balance_rows_ ? y_balanced_ : base::traction_.vec());
    } else if (balance_rows_) {
        CHKERRTHROW(PetscLogEventBegin(ev_mult_, 0, 0, 0, 0));
        CHKERRTHROW(MatMult(G_, S_->vec(), y_balanced_));
        CHKERRTHROW(PetscLogEventEnd(ev_mult_, 0, 0, 0, 0));
    } else {
        CHKERRTHROW(MatMult(G_, S_->vec(), base::traction_.vec()));
    }
    if (balance_rows_) {
        CHKERRTHROW(PetscLogEventBegin(ev_redistribute_, 0, 0, 0, 0));
        CHKERRTHROW(VecScatterBegin(to_fault_layout_, y_balanced_, base::traction_.vec(),
                                    INSERT_VALUES, SCATTER_FORWARD));
        CHKERRTHROW(VecScatterEnd(to_fault_layout_, y_balanced_, base::traction_.vec(),
                                  INSERT_VALUES, SCATTER_FORWARD));
        CHKERRTHROW(PetscLogEventEnd(ev_redistribute_, 0, 0, 0, 0));
    }
    CHKERRTHROW(VecAXPY(base::traction_.vec(), time, t_boundary_->vec()));
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
    GreensFunctionIndices ind(*this);
    int rank;
    PetscInt M, N, n_gf;
    MPI_Comm comm = base::comm();

    CHKERRTHROW(PetscPrintf(comm, "create_discrete_greens_function()\n"));
    MPI_Comm_rank(comm, &rank);

    // Rows are distributed in units of fault elements (the row block size). If balanced, every
    // rank gets about the same number of rows. Otherwise the rows live on the rank that owns the
    // fault element, which leaves ranks without fault elements idle in G_ * S_.
    {
        // How are the fault elements distributed? Friction law and state live on these ranks.
        int const has_fault = ind.num_local_elements > 0 ? 1 : 0;
        int ranks_with_fault = 0, size_;
        PetscInt max_local = 0, total = 0;
        MPI_Comm_size(comm, &size_);
        MPI_Allreduce(&has_fault, &ranks_with_fault, 1, MPI_INT, MPI_SUM, comm);
        MPI_Allreduce(&ind.num_local_elements, &max_local, 1, MPIU_INT, MPI_MAX, comm);
        MPI_Allreduce(&ind.num_local_elements, &total, 1, MPIU_INT, MPI_SUM, comm);
        CHKERRTHROW(PetscPrintf(comm,
                                "Fault elements: %" PetscInt_FMT
                                " on %d of %d ranks (at most %" PetscInt_FMT " per rank)\n",
                                total, ranks_with_fault, size_, max_local));
    }
    PetscInt m_local = ind.m;
    if (balance_rows_) {
        PetscInt num_elements = 0;
        int size;
        MPI_Comm_size(comm, &size);
        MPI_Allreduce(&ind.num_local_elements, &num_elements, 1, MPIU_INT, MPI_SUM, comm);
        m_local = (num_elements / size + (rank < num_elements % size ? 1 : 0)) * ind.m_bs;
    }
    CHKERRTHROW(MatCreateDense(comm, m_local, ind.n, PETSC_DECIDE, PETSC_DECIDE, nullptr, &G_));
    CHKERRTHROW(MatSetBlockSizes(G_, ind.m_bs, ind.n_bs));
    CHKERRTHROW(MatGetSize(G_, &M, &N));
    CHKERRTHROW(PetscPrintf(
        comm, "Green's function operator size: %" PetscInt_FMT " x %" PetscInt_FMT "\n", M, N));

    S_ = std::make_unique<PetscVector>(ind.slip_block_size, ind.num_local_elements, comm);
    t_boundary_ = std::make_unique<PetscVector>(ind.m_bs, ind.num_local_elements, comm);

    CHKERRTHROW(VecGetSize(S_->vec(), &n_gf));
    if (balance_rows_) {
        setup_row_redistribution();
    }
    return n_gf;
}

void SeasQDDiscreteGreenOperator::setup_row_redistribution() {
    MPI_Comm comm = base::comm();
    CHKERRTHROW(MatCreateVecs(G_, nullptr, &y_balanced_));

    // Row i of G_ and entry i of the traction vector have the same global index, only the
    // distribution over the ranks differs: every rank gets the entries of the traction vector
    // that it owns.
    PetscInt first, last;
    CHKERRTHROW(VecGetOwnershipRange(base::traction_.vec(), &first, &last));
    IS is;
    CHKERRTHROW(ISCreateStride(comm, last - first, first, 1, &is));
    CHKERRTHROW(VecScatterCreate(y_balanced_, is, base::traction_.vec(), is, &to_fault_layout_));
    CHKERRTHROW(ISDestroy(&is));
}

void SeasQDDiscreteGreenOperator::convert_to_single_precision() {
    MPI_Comm comm = base::comm();
    PetscInt m, N, lda;
    CHKERRTHROW(MatGetLocalSize(G_, &m, nullptr));
    CHKERRTHROW(MatGetSize(G_, nullptr, &N));
    CHKERRTHROW(MatDenseGetLDA(G_, &lda));

    // The local part of a parallel dense matrix holds the local rows and all columns,
    // column major with leading dimension lda.
    G_single_.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(N));
    PetscScalar const* a;
    CHKERRTHROW(MatDenseGetArrayRead(G_, &a));
    for (PetscInt j = 0; j < N; ++j) {
        for (PetscInt i = 0; i < m; ++i) {
            G_single_[static_cast<std::size_t>(j) * m + i] =
                static_cast<float>(a[static_cast<std::size_t>(j) * lda + i]);
        }
    }
    CHKERRTHROW(MatDenseRestoreArrayRead(G_, &a));
    single_m_ = m;
    single_N_ = N;
    CHKERRTHROW(VecScatterCreateToAll(S_->vec(), &gather_slip_, &slip_all_));

    // Accuracy check on a smooth test vector: compare with the double precision product.
    // (S_ is overwritten in every update, so it can be used as scratch space here.)
    PetscInt first, last;
    CHKERRTHROW(VecGetOwnershipRange(S_->vec(), &first, &last));
    PetscScalar* x;
    CHKERRTHROW(VecGetArray(S_->vec(), &x));
    for (PetscInt i = 0; i < last - first; ++i) {
        x[i] = std::sin(0.37 * static_cast<double>(first + i));
    }
    CHKERRTHROW(VecRestoreArray(S_->vec(), &x));
    Vec y_ref, y_sp;
    CHKERRTHROW(MatCreateVecs(G_, nullptr, &y_ref));
    CHKERRTHROW(VecDuplicate(y_ref, &y_sp));
    CHKERRTHROW(MatMult(G_, S_->vec(), y_ref));
    matmult_single_precision(y_sp);
    PetscReal norm_ref, norm_diff;
    CHKERRTHROW(VecNorm(y_ref, NORM_2, &norm_ref));
    CHKERRTHROW(VecAXPY(y_sp, -1.0, y_ref));
    CHKERRTHROW(VecNorm(y_sp, NORM_2, &norm_diff));
    CHKERRTHROW(VecDestroy(&y_ref));
    CHKERRTHROW(VecDestroy(&y_sp));
    CHKERRTHROW(PetscPrintf(comm,
                            "Green's function operator stored in single precision: relative "
                            "error of G * x for a test vector %.2e\n",
                            norm_ref > 0.0 ? norm_diff / norm_ref : norm_diff));

    // The double precision matrix is not needed any more
    CHKERRTHROW(MatDestroy(&G_));
}

void SeasQDDiscreteGreenOperator::matmult_single_precision(Vec y) {
    CHKERRTHROW(PetscLogEventBegin(ev_gather_, 0, 0, 0, 0));
    CHKERRTHROW(
        VecScatterBegin(gather_slip_, S_->vec(), slip_all_, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRTHROW(VecScatterEnd(gather_slip_, S_->vec(), slip_all_, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRTHROW(PetscLogEventEnd(ev_gather_, 0, 0, 0, 0));
    CHKERRTHROW(PetscLogEventBegin(ev_mult_, 0, 0, 0, 0));
    PetscScalar const* x;
    PetscScalar* result;
    CHKERRTHROW(VecGetArrayRead(slip_all_, &x));
    CHKERRTHROW(VecGetArray(y, &result));

    PetscInt const m = single_m_;
    std::fill(result, result + m, 0.0);
    float const* A = G_single_.data();
    for (PetscInt j = 0; j < single_N_; ++j) {
        double const xj = x[j];
        float const* column = A + static_cast<std::size_t>(j) * m;
        for (PetscInt i = 0; i < m; ++i) {
            result[i] += xj * static_cast<double>(column[i]);
        }
    }

    CHKERRTHROW(VecRestoreArray(y, &result));
    CHKERRTHROW(VecRestoreArrayRead(slip_all_, &x));
    CHKERRTHROW(PetscLogEventEnd(ev_mult_, 0, 0, 0, 0));
}

void SeasQDDiscreteGreenOperator::back_up_file(std::string file_to_backup) {
    int rank;
    MPI_Comm_rank(base::comm(), &rank);
    std::string new_filename = file_to_backup + ".tmp";
    std::string bu_filename = file_to_backup + ".bu";
    if (rank == 0) {
        if (fs::exists(file_to_backup)) {
            try {
                if (fs::exists(bu_filename)) {
                    fs::remove(bu_filename);
                }
                fs::rename(file_to_backup, bu_filename);
            } catch (fs::filesystem_error& e) {
                std::cerr << "Error moving file: " << e.what() << std::endl;
            }
        }
    }
    MPI_Barrier(base::comm());
    if (rank == 0) {
        try {
            fs::rename(new_filename, file_to_backup);
        } catch (fs::filesystem_error& e) {
            std::cerr << "Error moving file: " << e.what() << std::endl;
        }
    }
}

void SeasQDDiscreteGreenOperator::write_discrete_greens_operator(
    LocalSimplexMesh<DomainDimension> const& mesh, PetscInt current_gf, PetscInt n_gf) {
    PetscViewer v;
    PetscLogDouble t0, t1;
    int commsize;

    MPI_Comm_size(base::comm(), &commsize);

    CHKERRTHROW(PetscTime(&t0));
    std::string new_filename = gf_operator_filename_ + ".tmp";
    CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_), new_filename.c_str(),
                                      FILE_MODE_WRITE, &v));
    CHKERRTHROW(PetscViewerBinarySetUseMPIIO(v, PETSC_TRUE));
    {
        PetscInt commsize_checkpoint = (PetscInt)commsize;
        CHKERRTHROW(PetscViewerBinaryWrite(v, &commsize_checkpoint, 1, PETSC_INT));
    }

    CHKERRTHROW(PetscViewerBinaryWrite(v, &current_gf, 1, PETSC_INT));

    CHKERRTHROW(MatView(G_, v));

    CHKERRTHROW(PetscViewerDestroy(&v));

    back_up_file(gf_operator_filename_);

    CHKERRTHROW(PetscTime(&t1));

    CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                            "write_discrete_greens_operator():matrix %1.2e (sec)\n",
                            (double)(t1 - t0)));
    CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                            "  status: computed %" PetscInt_FMT " / pending %" PetscInt_FMT "\n",
                            current_gf, n_gf - current_gf));

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
    std::string new_filename = gf_facet_filename_ + ".tmp";
    CHKERRTHROW(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)G_), new_filename.c_str(),
                                      FILE_MODE_WRITE, &v));
    CHKERRTHROW(PetscViewerBinarySetUseMPIIO(v, PETSC_TRUE));
    CHKERRTHROW(ISView(is, v));
    CHKERRTHROW(PetscViewerDestroy(&v));
    back_up_file(gf_facet_filename_);
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
        CHKERRTHROW(PetscViewerBinarySetUseMPIIO(v, PETSC_TRUE));
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
SeasQDDiscreteGreenOperator ::create_row_col_permutation_matrices(bool create_row, bool create_col,
                                                                  bool row_layout_of_G) {
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

    // G_ may already have been converted to single precision and destroyed, in which case only
    // the layout of the fault elements is needed (to permute vectors)
    br = base::adapter().traction_block_size();
    bc = base::friction().slip_block_size();
    M = num_global_elements * br;
    N = num_global_elements * bc;
    n = num_local_elements * bc;
    if (G_ && row_layout_of_G) {
        CHKERRTHROW(MatGetLocalSize(G_, &m, nullptr));
    } else {
        m = num_local_elements * br;
    }

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
    CHKERRTHROW(PetscViewerBinarySetUseMPIIO(v, PETSC_TRUE));

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
    CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                            "  status: loaded %" PetscInt_FMT " / pending %" PetscInt_FMT "\n",
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
    GreensFunctionIndices ind(*this);

    PetscInt start = current_gf;
    PetscInt N = n_gf;
    Stopwatch sw;
    int rank;
    MPI_Comm comm = base::comm();

    if (start == N)
        return;

    if (gf_groups_ > 1 && worker_factory_) {
        partial_assemble_discrete_greens_function_grouped(mesh, start, N);
        return;
    }

    MPI_Comm_rank(comm, &rank);

    auto scatter = Scatter(base::adapter().fault_map().scatter_plan());
    auto ghost = scatter.template recv_prototype<double>(ind.slip_block_size, ALIGNMENT);

    if (start > 0) {
        CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                                "partial_assemble_discrete_greens_function() [%" PetscInt_FMT
                                " , %" PetscInt_FMT ")\n",
                                start, N));
    }
    double solve_time = 0.0;
    double solve_time_from_start = 0.0;
    for (PetscInt i = start; i < N; ++i) {
        std::string current_datetime = tndm::get_current_date_time_string();
        CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_),
                                "%s Computing Green's function %" PetscInt_FMT " / %" PetscInt_FMT
                                "\n",
                                current_datetime.c_str(), i, N));
        sw.start();
        CHKERRTHROW(VecZeroEntries(S_->vec()));
        if (i >= ind.nb_offset && i < ind.nb_offset + ind.m) {
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
        for (std::size_t faultNo = 0; faultNo < ind.num_local_elements; ++faultNo) {
            PetscInt g_m = ind.mb_offset + faultNo;
            PetscInt g_n = i;
            auto traction_block = traction_handle.subtensor(slice{}, faultNo);
            CHKERRTHROW(
                MatSetValuesBlocked(G_, 1, &g_m, 1, &g_n, traction_block.data(), INSERT_VALUES));
        }
        base::traction_.end_access_readonly(traction_handle);
        double step_time = sw.stop();
        solve_time += step_time;
        solve_time_from_start += step_time;

        double avg_time = solve_time_from_start / (i + 1 - start);
        double etl = avg_time * (N - i - 1);
        std::string remaining_time = tndm::format_time(etl);

        current_datetime = tndm::get_current_date_time_string();
        CHKERRTHROW(PetscPrintf(PetscObjectComm((PetscObject)G_), "%s (%s left)\n",
                                current_datetime.c_str(), remaining_time.c_str()));

        current_gf = i + 1;

        if (checkpoint_enabled_) {
            /* checkpoint */
            MPI_Bcast(&solve_time, 1, MPI_DOUBLE, 0, comm);
            if (solve_time / 60.0 > checkpoint_every_nmins_) {
                CHKERRTHROW(MatAssemblyBegin(G_, MAT_FINAL_ASSEMBLY));
                CHKERRTHROW(MatAssemblyEnd(G_, MAT_FINAL_ASSEMBLY));
                write_discrete_greens_operator(mesh, current_gf, n_gf);
                solve_time = 0.0;
            }
        }
    }

    CHKERRTHROW(MatAssemblyBegin(G_, MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd(G_, MAT_FINAL_ASSEMBLY));
}

/**
 * Computes the Green's functions [start, N) with several groups of ranks working concurrently.
 *
 * Every Green's function is the traction response to a unit slip at one slip degree of freedom,
 * and they are all independent. The ranks are split into `groups` groups of consecutive ranks.
 * Each group owns a complete copy of the discretisation (its own mesh partition, operators and
 * multigrid hierarchy) on its sub-communicator, which scales better than using all ranks for
 * a single solve.
 *
 * Layouts. G_ lives on the full communicator, numbered by the fault elements of the full
 * partition. A group numbers its fault elements differently, as its mesh is partitioned
 * differently. The fault facets are identified by their (global) vertex indices, which gives
 * the translation in both directions. Column j of G_ is the response to a unit slip at slip degree
 * of freedom j, i.e. component (j % slip_block_size) of fault element (j / slip_block_size).
 *
 * Work distribution. The groups proceed in lock-step rounds: in round r, group g computes column
 * start + r * groups + g (idle if that is beyond N). After each round G_ is assembled, which is
 * collective on all ranks and moves the rows to the ranks owning them in the full layout. So
 * after every round the columns [0, start + (r+1) * groups) are complete, which is exactly the
 * invariant the checkpoints rely on.
 */
void SeasQDDiscreteGreenOperator::partial_assemble_discrete_greens_function_grouped(
    LocalSimplexMesh<DomainDimension> const& mesh, PetscInt start, PetscInt N) {
    using facet_t = Simplex<DomainDimension - 1u>;
    constexpr int facet_size = DomainDimension;

    GreensFunctionIndices ind(*this);
    MPI_Comm comm = ind.comm;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int const groups = std::min(static_cast<int>(gf_groups_), size);
    if (groups < static_cast<int>(gf_groups_)) {
        CHKERRTHROW(PetscPrintf(comm,
                                "Warning: gf_groups = %u exceeds the number of ranks (%d), "
                                "using %d groups.\n",
                                gf_groups_, size, groups));
    }
    // Groups of consecutive ranks (which are typically on the same node)
    int const color = static_cast<int>((static_cast<long long>(rank) * groups) / size);

    struct CommGuard {
        MPI_Comm comm = MPI_COMM_NULL;
        ~CommGuard() {
            if (comm != MPI_COMM_NULL) {
                MPI_Comm_free(&comm);
            }
        }
    } group;
    MPI_Comm_split(comm, color, rank, &group.comm);

    CHKERRTHROW(PetscPrintf(comm,
                            "Computing Green's functions with %d groups of about %d ranks each\n",
                            groups, size / groups));

    auto allgather = [](MPI_Comm c, std::vector<PetscInt> const& local) {
        int csize;
        MPI_Comm_size(c, &csize);
        int const count = static_cast<int>(local.size());
        std::vector<int> counts(csize), displs(csize);
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, c);
        int total = 0;
        for (int i = 0; i < csize; ++i) {
            displs[i] = total;
            total += counts[i];
        }
        std::vector<PetscInt> all(total);
        MPI_Allgatherv(local.data(), count, MPIU_INT, all.data(), counts.data(), displs.data(),
                       MPIU_INT, c);
        return all;
    };

    // Index of every fault element in the full layout, by its facet. Elements of the full layout
    // are numbered rank by rank, which is the order of the gathered keys.
    auto const& fault_map = base::adapter().fault_map();
    auto const& all_facets = mesh.facets();
    PetscInt const num_local = static_cast<PetscInt>(fault_map.local_size());
    if (num_local != ind.num_local_elements) {
        throw std::logic_error("Number of local fault elements is inconsistent.");
    }
    std::vector<PetscInt> local_keys(num_local * facet_size);
    for (PetscInt b = 0; b < num_local; ++b) {
        auto const f = all_facets[fault_map.fctNo(b)];
        for (int d = 0; d < facet_size; ++d) {
            local_keys[b * facet_size + d] = static_cast<PetscInt>(f[d]);
        }
    }
    auto const all_keys = allgather(comm, local_keys);
    PetscInt const num_fault_elements = static_cast<PetscInt>(all_keys.size() / facet_size);
    if (num_fault_elements * ind.slip_block_size != N) {
        throw std::logic_error("Number of fault elements is inconsistent with the size of G.");
    }
    std::unordered_map<facet_t, PetscInt, SimplexHash<DomainDimension - 1u>> full_index(
        num_fault_elements);
    for (PetscInt n = 0; n < num_fault_elements; ++n) {
        facet_t plex;
        for (int d = 0; d < facet_size; ++d) {
            plex[d] = all_keys[n * facet_size + d];
        }
        full_index[plex] = n;
    }

    // Collective on the group: builds the mesh partition and the operators of this group.
    // (Declared after the communicator guard, so that it is destroyed before the communicator.)
    auto worker = worker_factory_(group.comm);
    GfSolver& solver = worker->solver();
    auto const& worker_facets = worker->fault_facets();
    if (static_cast<PetscInt>(solver.friction().slip_block_size()) != ind.slip_block_size ||
        static_cast<PetscInt>(solver.adapter().traction_block_size()) != ind.m_bs) {
        throw std::logic_error("Block sizes of the group solver differ from the full solver.");
    }

    // Translate between the layout of this group and the full layout.
    PetscInt const num_worker = static_cast<PetscInt>(worker_facets.size());
    std::vector<PetscInt> worker_to_full(num_worker);
    for (PetscInt b = 0; b < num_worker; ++b) {
        auto it = full_index.find(worker_facets[b]);
        if (it == full_index.end()) {
            throw std::runtime_error(
                "Fault facet of a Green's function group not found in the full mesh.");
        }
        worker_to_full[b] = it->second;
    }
    // Position of every fault element in the layout of this group
    auto const group_to_full = allgather(group.comm, worker_to_full);
    if (static_cast<PetscInt>(group_to_full.size()) != num_fault_elements) {
        throw std::runtime_error("Mesh of a Green's function group has a different number of "
                                 "fault elements than the full mesh.");
    }
    std::vector<PetscInt> full_to_group(num_fault_elements, -1);
    for (std::size_t s = 0; s < group_to_full.size(); ++s) {
        full_to_group[group_to_full[s]] = static_cast<PetscInt>(s);
    }
    if (std::find(full_to_group.begin(), full_to_group.end(), -1) != full_to_group.end()) {
        throw std::runtime_error("Fault elements of a Green's function group do not match the "
                                 "fault elements of the full mesh.");
    }

    if (start > 0) {
        CHKERRTHROW(PetscPrintf(comm,
                                "partial_assemble_discrete_greens_function() [%" PetscInt_FMT
                                " , %" PetscInt_FMT ")\n",
                                start, N));
    }

    PetscInt const rounds = (N - start + groups - 1) / groups;
    std::vector<PetscInt> rows(ind.m_bs);
    Stopwatch sw;
    double solve_time = 0.0;
    double solve_time_from_start = 0.0;
    for (PetscInt r = 0; r < rounds; ++r) {
        PetscInt const first = start + r * groups;
        PetscInt const last = std::min<PetscInt>(N, first + groups);
        CHKERRTHROW(PetscPrintf(comm,
                                "%s Computing Green's functions [%" PetscInt_FMT " , %" PetscInt_FMT
                                ") / %" PetscInt_FMT "\n",
                                tndm::get_current_date_time_string().c_str(), first, last, N));
        sw.start();

        PetscInt const j = first + color;
        if (j < N) {
            PetscInt const element = j / ind.slip_block_size;
            PetscInt const component = j % ind.slip_block_size;
            solver.unit_slip_response(full_to_group[element] * ind.slip_block_size + component);

            auto& traction = solver.traction();
            auto traction_handle = traction.begin_access_readonly();
            for (PetscInt b = 0; b < num_worker; ++b) {
                auto traction_block = traction_handle.subtensor(slice{}, b);
                for (PetscInt c = 0; c < ind.m_bs; ++c) {
                    rows[c] = worker_to_full[b] * ind.m_bs + c;
                }
                // Rows are in general owned by other ranks, they are sent when assembling
                CHKERRTHROW(MatSetValues(G_, ind.m_bs, rows.data(), 1, &j, traction_block.data(),
                                         INSERT_VALUES));
            }
            traction.end_access_readonly(traction_handle);
        }

        // Collective on all ranks: moves the rows to their owners
        CHKERRTHROW(MatAssemblyBegin(G_, MAT_FINAL_ASSEMBLY));
        CHKERRTHROW(MatAssemblyEnd(G_, MAT_FINAL_ASSEMBLY));

        // All ranks use the same time, so that the checkpointing decision below is identical.
        double const local_step_time = sw.stop();
        double step_time;
        MPI_Allreduce(&local_step_time, &step_time, 1, MPI_DOUBLE, MPI_MAX, comm);
        solve_time += step_time;
        solve_time_from_start += step_time;

        double const avg_time = solve_time_from_start / (r + 1);
        std::string const remaining_time = tndm::format_time(avg_time * (rounds - r - 1));
        CHKERRTHROW(PetscPrintf(comm, "%s (%s left)\n",
                                tndm::get_current_date_time_string().c_str(),
                                remaining_time.c_str()));

        if (checkpoint_enabled_ && solve_time / 60.0 > checkpoint_every_nmins_) {
            // G_ is assembled, columns [0, last) are complete
            write_discrete_greens_operator(mesh, last, N);
            solve_time = 0.0;
        }
    }
}

void SeasQDDiscreteGreenOperator::get_discrete_greens_function(
    LocalSimplexMesh<DomainDimension> const& mesh) {
    PetscBool found = PETSC_FALSE;
    PetscInt n_gfloaded = 0, n_gf;

    // Create an empty dense matrix to store all GFs
    if (!G_) {
        n_gf = create_discrete_greens_function();
    }

    if (checkpoint_enabled_) {
        // If a checkpoint file is found, load it. Record the number of assembled GFs found in file.
        CHKERRTHROW(PetscTestFile(gf_operator_filename_.c_str(), 'r', &found));
        if (found) {
            n_gfloaded = load_discrete_greens_operator(mesh, n_gf);
        }
    }

    // Assemble as many GFs as possible in the range [current_gf, n_gf)
    partial_assemble_discrete_greens_function(mesh, n_gfloaded, n_gf);

    if (checkpoint_enabled_) {
        // Write out the operator whenever the fully assembled operator was not loaded from file
        if (n_gfloaded != n_gf) {
            write_discrete_greens_operator(mesh, n_gf, n_gf);
        }
    }
}

void SeasQDDiscreteGreenOperator::write_discrete_greens_traction() {
    PetscViewer v;
    PetscLogDouble t0, t1;

    CHKERRTHROW(PetscTime(&t0));
    CHKERRTHROW(
        PetscViewerBinaryOpen(base::comm(), gf_traction_filename_.c_str(), FILE_MODE_WRITE, &v));
    CHKERRTHROW(VecView(t_boundary_->vec(), v));
    CHKERRTHROW(PetscViewerDestroy(&v));
    CHKERRTHROW(PetscTime(&t1));
    CHKERRTHROW(PetscPrintf(base::comm(), "write_discrete_greens_traction() %1.2e (sec)\n",
                            (double)(t1 - t0)));
}

void SeasQDDiscreteGreenOperator::load_discrete_greens_traction() {
    PetscViewer v;
    PetscLogDouble t0, t1;

    CHKERRTHROW(PetscTime(&t0));
    CHKERRTHROW(
        PetscViewerBinaryOpen(base::comm(), gf_traction_filename_.c_str(), FILE_MODE_READ, &v));
    CHKERRTHROW(VecLoad(base::traction_.vec(), v));
    CHKERRTHROW(PetscViewerDestroy(&v));
    CHKERRTHROW(PetscTime(&t1));
    CHKERRTHROW(PetscPrintf(base::comm(), "load_discrete_greens_operator() %1.2e (sec)\n",
                            (double)(t1 - t0)));

    if (repartition_gfs_) {
        Vec tmp;

        CHKERRTHROW(VecDuplicate(base::traction_.vec(), &tmp));
        auto [Rperm, Cperm] = create_row_col_permutation_matrices(true, false, false);
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
