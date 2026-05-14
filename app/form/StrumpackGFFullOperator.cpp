#include "StrumpackGFFullOperator.h"
#include "common/PetscUtil.h"
#include "hilbert.hpp"

#include <dense/DenseMatrix.hpp>
#include <misc/MPIWrapper.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace tndm {

// ---------------------------------------------------------------------------
// PCA helpers (identical to StrumpackGFOperator)
// ---------------------------------------------------------------------------

static void sym2_eigen(const double* A, double* eval, double* evec) {
    double a = A[0], b = A[1], d = A[3];
    double tr   = a + d;
    double disc = std::sqrt(std::max(0.0, 0.25*(a-d)*(a-d) + b*b));
    eval[0] = 0.5*tr + disc;
    eval[1] = 0.5*tr - disc;
    if (std::abs(b) > 1e-14) {
        double v0x = b, v0y = eval[0] - a;
        double len = std::sqrt(v0x*v0x + v0y*v0y);
        evec[0] =  v0x/len;  evec[1] = v0y/len;
        evec[2] = -v0y/len;  evec[3] = v0x/len;
    } else if (a >= d) {
        evec[0]=1; evec[1]=0; evec[2]=0; evec[3]=1;
    } else {
        evec[0]=0; evec[1]=1; evec[2]=1; evec[3]=0;
    }
}

static void sym3_eigen(const double* A, double* eval, double* evec) {
    double M[3][3], V[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) M[i][j] = A[i*3+j];
    for (int iter = 0; iter < 100; ++iter) {
        double maxv = 0.0; int p = 0, q = 1;
        for (int i=0; i<3; ++i)
            for (int j=i+1; j<3; ++j)
                if (std::abs(M[i][j]) > maxv) { maxv=std::abs(M[i][j]); p=i; q=j; }
        if (maxv < 1e-14) break;
        double tau = (M[q][q]-M[p][p]) / (2.0*M[p][q]);
        double t   = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau)+std::sqrt(1.0+tau*tau));
        double c   = 1.0/std::sqrt(1.0+t*t), s = t*c;
        double App=M[p][p], Aqq=M[q][q], Apq=M[p][q];
        M[p][p] = App - t*Apq;  M[q][q] = Aqq + t*Apq;  M[p][q]=M[q][p]=0.0;
        for (int r=0; r<3; ++r) {
            if (r==p||r==q) continue;
            double Mrp=M[r][p], Mrq=M[r][q];
            M[r][p]=M[p][r]= c*Mrp - s*Mrq;
            M[r][q]=M[q][r]= s*Mrp + c*Mrq;
        }
        for (int r=0; r<3; ++r) {
            double Vrp=V[r][p], Vrq=V[r][q];
            V[r][p]= c*Vrp - s*Vrq;
            V[r][q]= s*Vrp + c*Vrq;
        }
    }
    int idx[3]={0,1,2};
    if (M[idx[0]][idx[0]]<M[idx[1]][idx[1]]) std::swap(idx[0],idx[1]);
    if (M[idx[1]][idx[1]]<M[idx[2]][idx[2]]) std::swap(idx[1],idx[2]);
    if (M[idx[0]][idx[0]]<M[idx[1]][idx[1]]) std::swap(idx[0],idx[1]);
    for (int k=0; k<3; ++k) {
        eval[k] = M[idx[k]][idx[k]];
        for (int d=0; d<3; ++d) evec[k*3+d] = V[d][idx[k]];
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

StrumpackGFFullOperator::StrumpackGFFullOperator(
    Mat G_dense,
    const std::vector<PetscReal>& local_coords,
    PetscInt nbf, int D,
    Vec s_proto, Vec t_proto,
    MPI_Comm comm,
    HMatrixConfig const& config)
    : D_(D), slip_D_(D - 1), nbf_(nbf), comm_(comm), config_(config)
{
    static_assert(sizeof(PetscScalar) == sizeof(double),
                  "StrumpackGFFullOperator requires real PETSc scalar");

    int n_ranks, my_rank;
    CHKERRTHROW(MPI_Comm_size(comm_, &n_ranks));
    CHKERRTHROW(MPI_Comm_rank(comm_, &my_rank));

    PetscInt M_global, N_global;
    CHKERRTHROW(MatGetSize(G_dense, &M_global, &N_global));
    N_el_ = M_global / (D_ * nbf_);
    assert(N_el_ * D_      * nbf_ == M_global);
    assert(N_el_ * slip_D_ * nbf_ == N_global);

    const PetscInt Np = N_el_ * nbf_;

    // PETSc PETSC_DECIDE distributions for D*Np (rows) and slip_D*Np (cols)
    petsc_dist_row_.resize(n_ranks + 1, 0);
    petsc_dist_col_.resize(n_ranks + 1, 0);
    for (int r = 0; r < n_ranks; ++r) {
        petsc_dist_row_[r+1] = petsc_dist_row_[r] + D_*Np/n_ranks      + (r < (D_*Np)     %n_ranks ? 1 : 0);
        petsc_dist_col_[r+1] = petsc_dist_col_[r] + slip_D_*Np/n_ranks + (r < (slip_D_*Np)%n_ranks ? 1 : 0);
    }

    // Work vecs for MatMult inside construction callback
    const PetscInt local_col = petsc_dist_col_[my_rank+1] - petsc_dist_col_[my_rank];
    const PetscInt local_row = petsc_dist_row_[my_rank+1] - petsc_dist_row_[my_rank];
    CHKERRTHROW(VecCreateMPIWithArray(comm_, 1, local_col, slip_D_*Np, nullptr, &col_work_));
    CHKERRTHROW(VecCreateMPIWithArray(comm_, 1, local_row, D_*Np,      nullptr, &row_work_));

    // ---- Step 1: Allgather local coords → global (N_el*nbf × D) ----
    PetscInt local_n = static_cast<PetscInt>(local_coords.size() / D_);
    std::vector<PetscInt> all_n(n_ranks);
    CHKERRTHROW(MPI_Allgather(&local_n, 1, MPIU_INT, all_n.data(), 1, MPIU_INT, comm_));
    std::vector<int> rcounts(n_ranks), displs(n_ranks, 0);
    for (int r = 0; r < n_ranks; ++r) rcounts[r] = static_cast<int>(all_n[r]) * D_;
    for (int r = 1; r < n_ranks; ++r) displs[r] = displs[r-1] + rcounts[r-1];
    std::vector<PetscReal> global_coords(Np * D_);
    CHKERRTHROW(MPI_Allgatherv(local_coords.data(),
                               static_cast<int>(local_coords.size()), MPIU_REAL,
                               global_coords.data(), rcounts.data(), displs.data(),
                               MPIU_REAL, comm_));

    // ---- Step 2: Hilbert ordering on Np scalar node cloud ----
    build_spatial_permutation(global_coords);

    // ---- Step 3: Precompute global row/col index maps ----
    build_index_maps();

    // ---- Step 4: Build one STRUMPACK structured matrix ----
    build_s_full(G_dense);

    // ---- Step 5: Scatter objects for apply() ----
    build_scatters(s_proto, t_proto);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

StrumpackGFFullOperator::~StrumpackGFFullOperator() {
    VecDestroy(&col_work_);
    VecDestroy(&row_work_);
    VecDestroy(&s_perm_work_);
    VecDestroy(&t_perm_work_);
    VecScatterDestroy(&scatter_s_);
    VecScatterDestroy(&scatter_t_);
}

// ---------------------------------------------------------------------------
// build_spatial_permutation — PCA-Hilbert ordering (identical to StrumpackGFOperator)
// ---------------------------------------------------------------------------

void StrumpackGFFullOperator::build_spatial_permutation(
    const std::vector<PetscReal>& global_coords)
{
    const PetscInt Np = N_el_ * nbf_;
    perm_.resize(Np);
    std::iota(perm_.begin(), perm_.end(), 0);

    std::vector<double> mean(D_, 0.0);
    for (PetscInt i = 0; i < Np; ++i)
        for (int d = 0; d < D_; ++d)
            mean[d] += static_cast<double>(global_coords[i*D_+d]);
    for (int d = 0; d < D_; ++d) mean[d] /= static_cast<double>(Np);

    std::vector<double> cov(D_*D_, 0.0);
    for (PetscInt i = 0; i < Np; ++i)
        for (int a = 0; a < D_; ++a) {
            double va = static_cast<double>(global_coords[i*D_+a]) - mean[a];
            for (int b = a; b < D_; ++b)
                cov[a*D_+b] += va * (static_cast<double>(global_coords[i*D_+b]) - mean[b]);
        }
    for (int a = 0; a < D_; ++a)
        for (int b = a+1; b < D_; ++b)
            cov[b*D_+a] = cov[a*D_+b];

    std::vector<double> eval(D_), evec(D_*D_);
    if (D_ == 2) sym2_eigen(cov.data(), eval.data(), evec.data());
    else         sym3_eigen(cov.data(), eval.data(), evec.data());

    const double thr = static_cast<double>(config_.leaf_size) / static_cast<double>(Np);
    int eff_dim = 0;
    for (int d = 0; d < D_; ++d)
        if (eval[0] > 0.0 && eval[d] / eval[0] > thr) ++eff_dim;
    if (eff_dim < 1) eff_dim = 1;

    {
        int rank; MPI_Comm_rank(comm_, &rank);
        if (rank == 0) {
            std::cout << "  Hilbert ordering: eff_dim=" << eff_dim << "  eigenvalue ratios [";
            for (int d = 0; d < D_; ++d) {
                std::cout << (eval[0]>0.0 ? eval[d]/eval[0] : 0.0);
                if (d < D_-1) std::cout << ", ";
            }
            std::cout << "]  threshold=" << thr << "\n";
        }
    }

    std::vector<double> proj(Np * eff_dim);
    for (PetscInt i = 0; i < Np; ++i)
        for (int k = 0; k < eff_dim; ++k) {
            double p = 0.0;
            for (int d = 0; d < D_; ++d)
                p += (static_cast<double>(global_coords[i*D_+d]) - mean[d]) * evec[k*D_+d];
            proj[i*eff_dim+k] = p;
        }

    std::vector<double> pmin(eff_dim,  std::numeric_limits<double>::max());
    std::vector<double> pmax(eff_dim, -std::numeric_limits<double>::max());
    for (PetscInt i = 0; i < Np; ++i)
        for (int k = 0; k < eff_dim; ++k) {
            pmin[k] = std::min(pmin[k], proj[i*eff_dim+k]);
            pmax[k] = std::max(pmax[k], proj[i*eff_dim+k]);
        }
    double L = 0.0;
    for (int k = 0; k < eff_dim; ++k) L = std::max(L, pmax[k]-pmin[k]);
    if (L < 1e-14) L = 1.0;

    constexpr double GMAX = static_cast<double>(std::numeric_limits<uint16_t>::max());
    auto norm = [&](int i, int k) -> uint16_t {
        double t = (proj[i*eff_dim+k] - pmin[k]) / L;
        return static_cast<uint16_t>(std::min(t * GMAX, GMAX));
    };

    if (eff_dim == 1) {
        std::sort(perm_.begin(), perm_.end(),
                  [&](PetscInt a, PetscInt b){ return proj[a] < proj[b]; });
    } else if (eff_dim == 2) {
        using HIdx = std::array<uint16_t, 2>;
        std::vector<HIdx> codes(Np);
        for (PetscInt i = 0; i < Np; ++i)
            codes[i] = hilbert::v2::PositionToIndex(HIdx{ norm(i,0), norm(i,1) });
        std::sort(perm_.begin(), perm_.end(),
                  [&codes](PetscInt a, PetscInt b){ return codes[a] < codes[b]; });
    } else {
        using HIdx = std::array<uint16_t, 3>;
        std::vector<HIdx> codes(Np);
        for (PetscInt i = 0; i < Np; ++i)
            codes[i] = hilbert::v2::PositionToIndex(HIdx{ norm(i,0), norm(i,1), norm(i,2) });
        std::sort(perm_.begin(), perm_.end(),
                  [&codes](PetscInt a, PetscInt b){ return codes[a] < codes[b]; });
    }
}

// ---------------------------------------------------------------------------
// build_index_maps
// ---------------------------------------------------------------------------

void StrumpackGFFullOperator::build_index_maps() {
    const PetscInt Np = N_el_ * nbf_;

    row_perm_to_tandem_.resize(D_ * Np);
    for (PetscInt new_i = 0; new_i < Np; ++new_i) {
        PetscInt old_i = perm_[new_i];
        PetscInt e = old_i / nbf_, n = old_i % nbf_;
        for (int alpha = 0; alpha < D_; ++alpha)
            row_perm_to_tandem_[new_i * D_ + alpha] = e * D_ * nbf_ + alpha * nbf_ + n;
    }

    col_perm_to_tandem_.resize(slip_D_ * Np);
    for (PetscInt new_j = 0; new_j < Np; ++new_j) {
        PetscInt old_j = perm_[new_j];
        PetscInt e = old_j / nbf_, n = old_j % nbf_;
        for (int beta = 0; beta < slip_D_; ++beta)
            col_perm_to_tandem_[new_j * slip_D_ + beta] = e * slip_D_ * nbf_ + beta * nbf_ + n;
    }
}

// ---------------------------------------------------------------------------
// build_petsc_tree (identical to StrumpackGFOperator)
// ---------------------------------------------------------------------------

strumpack::structured::ClusterTree
StrumpackGFFullOperator::build_petsc_tree(const std::vector<int>& dist, int lo, int hi) {
    strumpack::structured::ClusterTree t(dist[hi] - dist[lo]);
    if (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        t.c.push_back(build_petsc_tree(dist, lo, mid));
        t.c.push_back(build_petsc_tree(dist, mid, hi));
    }
    return t;
}

// ---------------------------------------------------------------------------
// build_s_full — one STRUMPACK HODLR matrix, padded to D*Np x D*Np (square)
//
// The physical GF is D*Np x slip_D*Np (rectangular).  HODLR requires a square
// matrix, so we pad the column space by (D - slip_D)*Np fictitious "normal
// slip" columns that always produce zero traction output.  STRUMPACK probes
// these zero columns and compresses their off-diagonal blocks to rank 0.
//
// Column layout (permuted spatial order):
//   pc ∈ [0,            slip_D*Np) : real slip, col_perm_to_tandem_[pc] → G_dense col
//   pc ∈ [slip_D*Np,    D*Np     ) : zero padding — no MatMult contribution
// ---------------------------------------------------------------------------

void StrumpackGFFullOperator::build_s_full(Mat G_dense) {
    using namespace strumpack;
    using namespace strumpack::structured;

    int n_ranks, my_rank;
    CHKERRTHROW(MPI_Comm_size(comm_, &n_ranks));
    CHKERRTHROW(MPI_Comm_rank(comm_, &my_rank));

    const PetscInt Np     = N_el_ * nbf_;
    const PetscInt M_tot  = D_     * Np;   // rows
    const PetscInt N_tot  = D_     * Np;   // cols (padded to square)
    const PetscInt N_real = slip_D_ * Np;  // real (non-zero) cols

    const PetscInt local_row = petsc_dist_row_[my_rank+1] - petsc_dist_row_[my_rank];
    const PetscInt local_col = petsc_dist_col_[my_rank+1] - petsc_dist_col_[my_rank];

    // allgatherv params — row space (D*Np) used for both rows AND padded cols
    std::vector<int> row_counts(n_ranks), row_displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) row_counts[r] = petsc_dist_row_[r+1] - petsc_dist_row_[r];
    row_displs[0] = 0;
    for (int r = 1; r < n_ranks; ++r) row_displs[r] = petsc_dist_row_[r];

    // allgatherv params — G_dense col space (slip_D*Np) for adj MatMultTranspose output
    std::vector<int> col_counts(n_ranks), col_displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) col_counts[r] = petsc_dist_col_[r+1] - petsc_dist_col_[r];
    col_displs[0] = 0;
    for (int r = 1; r < n_ranks; ++r) col_displs[r] = petsc_dist_col_[r];

    // Scratch buffers (all D*Np-sized to cover the padded square space)
    std::vector<double> x_full(M_tot);          // allgather of STRUMPACK input
    std::vector<double> x_tandem_slip(N_real);  // fwd: reordered slip input for MatMult
    std::vector<double> x_tandem_trac(M_tot);   // adj: reordered traction input for MatMultTranspose
    std::vector<double> y_full_row(M_tot);      // fwd: allgather of MatMult output (D*Np)
    std::vector<double> y_full_col(N_real);     // adj: allgather of MatMultTranspose output (slip_D*Np)
    std::vector<double> y_local_row(local_row);
    std::vector<double> y_local_col(local_col);

    auto& mvc_ref = matvec_count_;
    auto& adc_ref = adjoint_count_;
    MPI_Comm comm_cap = comm_;
    std::size_t cb_fwd = 0, cb_adj = 0;

    auto Amult = [&mvc_ref, &adc_ref,
                  &cb_fwd, &cb_adj,
                  &x_full,
                  &x_tandem_slip, &x_tandem_trac,
                  &y_full_row, &y_full_col,
                  &y_local_row, &y_local_col,
                  &row_counts, &row_displs,
                  &col_counts, &col_displs,
                  this, G_dense, n_ranks, my_rank, N_real, comm_cap]
        (Trans t,
         const DenseMatrix<double>& R,
         DenseMatrix<double>& S,
         const std::vector<int>& rdist,
         const std::vector<int>& cdist) {

        const int nvec = static_cast<int>(R.cols());
        if (t == Trans::N) mvc_ref += static_cast<std::size_t>(nvec);
        else               adc_ref += static_cast<std::size_t>(nvec);

        std::size_t& cb = (t == Trans::N) ? cb_fwd : cb_adj;
        ++cb;
        if (my_rank == 0)
            std::cout << "    [full " << (t == Trans::N ? "Gv  " : "G^Tw")
                      << " #" << cb << "] nvec=" << nvec
                      << "  fwd=" << mvc_ref.load()
                      << " adj=" << adc_ref.load() << "\n" << std::flush;

        const bool fwd = (t == Trans::N);

        // Both forward input and adjoint input live in the D*Np square space,
        // so the same row_counts/row_displs allgatherv applies to both.
        const std::vector<int>& in_dist = fwd ? cdist : rdist;
        std::vector<int> in_counts(n_ranks), in_displs(n_ranks);
        for (int r = 0; r < n_ranks; ++r) in_counts[r] = in_dist[r+1] - in_dist[r];
        in_displs[0] = 0;
        for (int r = 1; r < n_ranks; ++r) in_displs[r] = in_dist[r];

        const std::vector<int>& out_dist = fwd ? rdist : cdist;
        const int strumpack_out_start = out_dist[my_rank];
        const int strumpack_out_local = out_dist[my_rank+1] - out_dist[my_rank];

        for (int k = 0; k < nvec; ++k) {
            // Step 1: allgather STRUMPACK input → x_full (D*Np, permuted order)
            CHKERRTHROW(MPI_Allgatherv(
                R.ptr(0, k), in_counts[my_rank], MPI_DOUBLE,
                x_full.data(), in_counts.data(), in_displs.data(), MPI_DOUBLE, comm_cap));

            if (fwd) {
                // Step 2 (fwd): reorder real slip portion (pc < N_real) → x_tandem_slip
                //               padding portion (pc >= N_real) contributes zero → skip
                for (PetscInt pc = 0; pc < N_real; ++pc)
                    x_tandem_slip[col_perm_to_tandem_[pc]] = x_full[pc];

                // Step 3 (fwd): MatMult(G_dense, slip, traction)
                CHKERRTHROW(VecPlaceArray(col_work_, x_tandem_slip.data() + petsc_dist_col_[my_rank]));
                CHKERRTHROW(VecPlaceArray(row_work_, y_local_row.data()));
                CHKERRTHROW(MatMult(G_dense, col_work_, row_work_));
                CHKERRTHROW(VecResetArray(col_work_));
                CHKERRTHROW(VecResetArray(row_work_));

                // Step 4 (fwd): allgather MatMult output (D*Np, Tandem order) → y_full_row
                CHKERRTHROW(MPI_Allgatherv(
                    y_local_row.data(), row_counts[my_rank], MPI_DOUBLE,
                    y_full_row.data(), row_counts.data(), row_displs.data(), MPI_DOUBLE, comm_cap));

                // Step 5 (fwd): reorder Tandem → permuted row, write STRUMPACK output slice
                for (int i = 0; i < strumpack_out_local; ++i)
                    S.ptr(0, k)[i] = y_full_row[row_perm_to_tandem_[strumpack_out_start + i]];

            } else {
                // Step 2 (adj): reorder full row input → x_tandem_trac
                for (PetscInt pr = 0; pr < static_cast<PetscInt>(x_full.size()); ++pr)
                    x_tandem_trac[row_perm_to_tandem_[pr]] = x_full[pr];

                // Step 3 (adj): MatMultTranspose(G_dense, traction, slip)
                CHKERRTHROW(VecPlaceArray(row_work_, x_tandem_trac.data() + petsc_dist_row_[my_rank]));
                CHKERRTHROW(VecPlaceArray(col_work_, y_local_col.data()));
                CHKERRTHROW(MatMultTranspose(G_dense, row_work_, col_work_));
                CHKERRTHROW(VecResetArray(row_work_));
                CHKERRTHROW(VecResetArray(col_work_));

                // Step 4 (adj): allgather MatMultTranspose output (slip_D*Np) → y_full_col
                CHKERRTHROW(MPI_Allgatherv(
                    y_local_col.data(), col_counts[my_rank], MPI_DOUBLE,
                    y_full_col.data(), col_counts.data(), col_displs.data(), MPI_DOUBLE, comm_cap));

                // Step 5 (adj): reorder permuted col → Tandem, write STRUMPACK output slice
                //               entries in the padding region (po >= N_real) → zero
                for (int i = 0; i < strumpack_out_local; ++i) {
                    const int po = strumpack_out_start + i;
                    S.ptr(0, k)[i] = (po < N_real)
                                     ? y_full_col[col_perm_to_tandem_[po]]
                                     : 0.0;
                }
            }
        }
    };

    // Square matrix: both row and col trees use the D*Np distribution
    ClusterTree row_tree = build_petsc_tree(petsc_dist_row_, 0, n_ranks);

    StructuredOptions<double> opts(Type::HODLR);
    opts.set_rel_tol(config_.rtol);
    opts.set_leaf_size(config_.leaf_size);
    opts.set_max_rank(config_.max_rank);

    if (my_rank == 0)
        std::cout << "  Building S_full  (" << M_tot << " x " << N_tot
                  << " padded-square, " << N_real << " real cols"
                  << ", node-interleaved Hilbert order)\n";

    S_full_ = construct_matrix_free<double>(
        MPIComm(comm_), static_cast<int>(M_tot), static_cast<int>(N_tot),
        Amult, opts, &row_tree);

    mem_bytes_ = static_cast<double>(S_full_->memory());
}

// ---------------------------------------------------------------------------
// build_scatters
// ---------------------------------------------------------------------------

void StrumpackGFFullOperator::build_scatters(Vec s_proto, Vec t_proto) {
    int my_rank;
    CHKERRTHROW(MPI_Comm_rank(comm_, &my_rank));
    const PetscInt Np = N_el_ * nbf_;

    const PetscInt perm_cstart = petsc_dist_col_[my_rank];
    const PetscInt perm_cend   = petsc_dist_col_[my_rank + 1];
    const PetscInt local_col   = perm_cend - perm_cstart;

    const PetscInt perm_rstart = petsc_dist_row_[my_rank];
    const PetscInt perm_rend   = petsc_dist_row_[my_rank + 1];
    const PetscInt local_row   = perm_rend - perm_rstart;

    CHKERRTHROW(VecCreateMPI(comm_, local_col, slip_D_ * Np, &s_perm_work_));
    CHKERRTHROW(VecCreateMPI(comm_, local_row, D_     * Np, &t_perm_work_));

    // scatter_s_: s (Tandem slip) → s_perm_work_  (permuted col order)
    {
        std::vector<PetscInt> from_idx(local_col);
        for (PetscInt pc = perm_cstart; pc < perm_cend; ++pc)
            from_idx[pc - perm_cstart] = col_perm_to_tandem_[pc];
        IS from_IS, to_IS;
        CHKERRTHROW(ISCreateGeneral(comm_, local_col, from_idx.data(), PETSC_COPY_VALUES, &from_IS));
        CHKERRTHROW(ISCreateStride (comm_, local_col, perm_cstart, 1, &to_IS));
        CHKERRTHROW(VecScatterCreate(s_proto, from_IS, s_perm_work_, to_IS, &scatter_s_));
        CHKERRTHROW(ISDestroy(&from_IS));
        CHKERRTHROW(ISDestroy(&to_IS));
    }

    // scatter_t_: t_perm_work_ (permuted row order) → t (Tandem traction, ADD)
    {
        std::vector<PetscInt> to_idx(local_row);
        for (PetscInt pr = perm_rstart; pr < perm_rend; ++pr)
            to_idx[pr - perm_rstart] = row_perm_to_tandem_[pr];
        IS from_IS, to_IS;
        CHKERRTHROW(ISCreateStride (comm_, local_row, perm_rstart, 1, &from_IS));
        CHKERRTHROW(ISCreateGeneral(comm_, local_row, to_idx.data(), PETSC_COPY_VALUES, &to_IS));
        CHKERRTHROW(VecScatterCreate(t_perm_work_, from_IS, t_proto, to_IS, &scatter_t_));
        CHKERRTHROW(ISDestroy(&from_IS));
        CHKERRTHROW(ISDestroy(&to_IS));
    }
}

// ---------------------------------------------------------------------------
// apply — t += S_full * s  (both in original Tandem DOF order)
// ---------------------------------------------------------------------------

void StrumpackGFFullOperator::apply(Vec s, Vec t) const {
    const PetscInt Np = N_el_ * nbf_;

    int n_ranks, my_rank;
    CHKERRTHROW(MPI_Comm_size(comm_, &n_ranks));
    CHKERRTHROW(MPI_Comm_rank(comm_, &my_rank));

    // Step 1: s (Tandem) → s_perm_work_ (permuted col order, PETSc dist)
    CHKERRTHROW(VecScatterBegin(scatter_s_, s, s_perm_work_, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRTHROW(VecScatterEnd  (scatter_s_, s, s_perm_work_, INSERT_VALUES, SCATTER_FORWARD));

    // Step 2: allgather full s_perm from PETSc's col distribution
    std::vector<int> col_counts(n_ranks), col_displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) col_counts[r] = petsc_dist_col_[r+1] - petsc_dist_col_[r];
    col_displs[0] = 0;
    for (int r = 1; r < n_ranks; ++r) col_displs[r] = petsc_dist_col_[r];

    // Gather s_perm (slip_D*Np) then build the padded D*Np input vector.
    // Real cols [0, slip_D*Np) come from s_perm; padding cols [slip_D*Np, D*Np) are zero.
    const PetscInt N_real = slip_D_ * Np;
    std::vector<double> x_perm_padded(D_ * Np, 0.0);  // zero-initialise covers padding
    {
        const PetscScalar* s_arr;
        CHKERRTHROW(VecGetArrayRead(s_perm_work_, &s_arr));
        CHKERRTHROW(MPI_Allgatherv(s_arr, col_counts[my_rank], MPI_DOUBLE,
                                   x_perm_padded.data(), col_counts.data(),
                                   col_displs.data(), MPI_DOUBLE, comm_));
        CHKERRTHROW(VecRestoreArrayRead(s_perm_work_, &s_arr));
    }

    // Step 3: extract STRUMPACK's local input slice from the padded vector, call mult
    const auto& cdist = S_full_->cdist();
    const auto& rdist = S_full_->rdist();
    const int s_in_local  = cdist[my_rank+1] - cdist[my_rank];
    const int s_out_local = rdist[my_rank+1] - rdist[my_rank];

    strumpack::DenseMatrix<double> x_dm(s_in_local, 1);
    std::memcpy(x_dm.data(), x_perm_padded.data() + cdist[my_rank], s_in_local * sizeof(double));
    (void)N_real;

    strumpack::DenseMatrix<double> y_dm(s_out_local, 1);
    S_full_->mult(strumpack::Trans::N, x_dm, y_dm);

    // Step 4: allgather S_full output from STRUMPACK's rdist → full y_perm
    std::vector<int> rdist_counts(n_ranks), rdist_displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) rdist_counts[r] = rdist[r+1] - rdist[r];
    rdist_displs[0] = 0;
    for (int r = 1; r < n_ranks; ++r) rdist_displs[r] = rdist[r];

    std::vector<double> y_perm_full(D_ * Np);
    CHKERRTHROW(MPI_Allgatherv(y_dm.data(), s_out_local, MPI_DOUBLE,
                               y_perm_full.data(), rdist_counts.data(),
                               rdist_displs.data(), MPI_DOUBLE, comm_));

    // Step 5: copy PETSc's local slice into t_perm_work_
    const PetscInt perm_rstart = petsc_dist_row_[my_rank];
    const PetscInt perm_rend   = petsc_dist_row_[my_rank + 1];
    {
        PetscScalar* t_arr;
        CHKERRTHROW(VecGetArray(t_perm_work_, &t_arr));
        for (PetscInt pr = perm_rstart; pr < perm_rend; ++pr)
            t_arr[pr - perm_rstart] = static_cast<PetscScalar>(y_perm_full[pr]);
        CHKERRTHROW(VecRestoreArray(t_perm_work_, &t_arr));
    }

    // Step 6: t_perm_work_ → t (Tandem traction, ADD)
    CHKERRTHROW(VecScatterBegin(scatter_t_, t_perm_work_, t, ADD_VALUES, SCATTER_FORWARD));
    CHKERRTHROW(VecScatterEnd  (scatter_t_, t_perm_work_, t, ADD_VALUES, SCATTER_FORWARD));
}

} // namespace tndm
