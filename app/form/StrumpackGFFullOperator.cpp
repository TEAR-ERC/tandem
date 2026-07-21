#include "StrumpackGFFullOperator.h"
#include "common/PetscUtil.h"
#include "hilbert.hpp"

#include <HODLR/HODLRMatrix.hpp>
#include <HODLR/HODLROptions.hpp>
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

    // Query PETSc's actual parallel layout for G_dense (rows)
    PetscInt G_rstart, G_rend;
    CHKERRTHROW(MatGetOwnershipRange(G_dense, &G_rstart, &G_rend));
    PetscInt G_local_rows = G_rend - G_rstart;

    std::vector<PetscInt> all_local_rows(n_ranks);
    CHKERRTHROW(MPI_Allgather(&G_local_rows, 1, MPIU_INT, all_local_rows.data(), 1, MPIU_INT, comm_));

    petsc_dist_row_.resize(n_ranks + 1, 0);
    for (int r = 0; r < n_ranks; ++r) {
        petsc_dist_row_[r+1] = petsc_dist_row_[r] + all_local_rows[r];
    }

    // Query PETSc's default parallel layout for slip_D*Np (cols)
    Vec temp_col;
    CHKERRTHROW(VecCreateMPI(comm_, PETSC_DECIDE, slip_D_ * Np, &temp_col));
    PetscInt col_rstart, col_rend;
    CHKERRTHROW(VecGetOwnershipRange(temp_col, &col_rstart, &col_rend));
    PetscInt col_local_size = col_rend - col_rstart;

    std::vector<PetscInt> all_local_cols(n_ranks);
    CHKERRTHROW(MPI_Allgather(&col_local_size, 1, MPIU_INT, all_local_cols.data(), 1, MPIU_INT, comm_));

    petsc_dist_col_.resize(n_ranks + 1, 0);
    for (int r = 0; r < n_ranks; ++r) {
        petsc_dist_col_[r+1] = petsc_dist_col_[r] + all_local_cols[r];
    }
    CHKERRTHROW(VecDestroy(&temp_col));

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

    // k-d tree path: the tree induces perm_ (spatially-compact contiguous ranges at every
    // level, down to leaf_size DOFs) and its depth is a function of leaf_size only, so the
    // resulting hierarchy — and hence the compression — is independent of the rank count.
    // For eff_dim==1 the median split degenerates to the same sorted order as Hilbert.
    if (config_.cluster_tree == "kdtree") {
        // leaf_size is a DOF count; the tree recurses over nodes, D DOFs each.
        const int node_leaf = std::max(1, config_.leaf_size / D_);
        std::vector<PetscInt> indices(Np);
        std::iota(indices.begin(), indices.end(), PetscInt{0});
        PetscInt fill = 0;
        kd_tree_ = std::make_unique<strumpack::structured::ClusterTree>(
            build_kdtree(indices, proj, eff_dim, node_leaf, fill));
        assert(fill == Np);
        return;
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

    // Padded column space, node-interleaved: D slots per node, of which the first
    // slip_D are real slip components and the last is fictitious (-1). Sharing the
    // stride D with the row map is what makes the square matrix's diagonal align
    // with the physical node-to-itself interaction.
    col_perm_to_tandem_.assign(D_ * Np, -1);
    for (PetscInt new_j = 0; new_j < Np; ++new_j) {
        PetscInt old_j = perm_[new_j];
        PetscInt e = old_j / nbf_, n = old_j % nbf_;
        for (int beta = 0; beta < slip_D_; ++beta)
            col_perm_to_tandem_[new_j * D_ + beta] = e * slip_D_ * nbf_ + beta * nbf_ + n;
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
// report_tree — E1 diagnostic: dump the imposed hierarchy on rank 0.
//
// The two numbers that matter:
//   levels    — with "petsc1d" this is log2(n_ranks)+1, i.e. the matrix structure is a
//               function of the job launch parameters. With "kdtree" it is set by
//               leaf_size and must not move when the rank count changes.
//   all_pad   — leaves containing no real column. Such a leaf is an exactly singular
//               diagonal block and detonates the randomized construction. Node-interleaved
//               padding makes this impossible; the count is printed to keep it honest.
// ---------------------------------------------------------------------------

static void collect_leaves(const strumpack::structured::ClusterTree& t, int offset,
                           std::vector<std::pair<int,int>>& leaves) {
    if (t.c.empty()) { leaves.emplace_back(offset, t.size); return; }
    for (const auto& ch : t.c) { collect_leaves(ch, offset, leaves); offset += ch.size; }
}

static void report_tree(const strumpack::structured::ClusterTree& tree,
                        int D, int slip_D, const char* label, int n_ranks) {
    std::vector<std::pair<int,int>> leaves;
    collect_leaves(tree, 0, leaves);

    int lmin = std::numeric_limits<int>::max(), lmax = 0, all_pad = 0;
    for (const auto& [start, len] : leaves) {
        lmin = std::min(lmin, len);
        lmax = std::max(lmax, len);
        bool has_real = false;
        for (int i = start; i < start + len && !has_real; ++i)
            if ((i % D) != slip_D) has_real = true;
        if (!has_real) ++all_pad;
    }
    if (leaves.empty()) { lmin = 0; }

    std::cout << "  row_tree[" << label << "]: levels=" << tree.levels()
              << " nleaves=" << leaves.size()
              << " min/max leaf=" << lmin << "/" << lmax
              << " all-padding leaves=" << all_pad
              << "  (n_ranks=" << n_ranks << ")\n";
    if (all_pad > 0)
        std::cout << "  *** WARNING: " << all_pad << " cluster-tree leaf/leaves contain no "
                  << "real column — these diagonal blocks are exactly singular ***\n";
    std::cout << std::flush;
}

// ---------------------------------------------------------------------------
// build_kdtree — spatial median-split ClusterTree that INDUCES perm_
//
// Ported from StrumpackGFOperator. Each leaf writes its node indices contiguously into
// perm_ (left subtree entirely before right), so every tree node covers a contiguous
// perm_ range AND a spatially compact tile — tree<->perm_ consistency by construction.
//
// Difference from the scalar version: this operator's index space is D DOFs per node
// (rows AND columns, thanks to the interleaved padding), so every ClusterTree node
// reports n*D rather than n.
// ---------------------------------------------------------------------------

strumpack::structured::ClusterTree
StrumpackGFFullOperator::build_kdtree(std::vector<PetscInt>& indices,
                                      const std::vector<double>& proj, int eff_dim,
                                      int leaf_size, PetscInt& fill) {
    const std::size_t n = indices.size();
    strumpack::structured::ClusterTree node(static_cast<int>(n) * D_);

    if (static_cast<int>(n) <= leaf_size) {
        for (PetscInt idx : indices) perm_[fill++] = idx;   // leaf: emit in place, no children
        return node;
    }

    // Longest spatial axis (largest extent of the projected coordinates).
    int dim = 0;
    double best = -1.0;
    for (int d = 0; d < eff_dim; ++d) {
        double lo =  std::numeric_limits<double>::max();
        double hi = -std::numeric_limits<double>::max();
        for (PetscInt idx : indices) {
            double v = proj[static_cast<std::size_t>(idx) * eff_dim + d];
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        if (hi - lo > best) { best = hi - lo; dim = d; }
    }

    // Balanced median split along `dim` (nth_element reorders `indices` in place).
    const std::size_t mid = n / 2;
    std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
                     [&](PetscInt a, PetscInt b) {
                         return proj[static_cast<std::size_t>(a) * eff_dim + dim] <
                                proj[static_cast<std::size_t>(b) * eff_dim + dim];
                     });

    std::vector<PetscInt> L(indices.begin(), indices.begin() + mid);
    std::vector<PetscInt> R(indices.begin() + mid, indices.end());
    node.c.push_back(build_kdtree(L, proj, eff_dim, leaf_size, fill));  // fills perm_ left-first
    node.c.push_back(build_kdtree(R, proj, eff_dim, leaf_size, fill));  // then right
    return node;
}

// ---------------------------------------------------------------------------
// build_s_full — one STRUMPACK HODLR matrix, padded to D*Np x D*Np (square)
//
// The physical GF is D*Np x slip_D*Np (rectangular).  HODLR requires a square
// matrix, so we pad the column space by (D - slip_D)*Np fictitious "normal
// slip" columns.
//
// Column layout (permuted spatial order), node-interleaved to match the rows:
//   pc = new_j*D + beta,  beta ∈ [0, slip_D) : real slip, col_perm_to_tandem_[pc] → G col
//   pc = new_j*D + slip_D                    : fictitious, carries only pad_scale_ on
//                                              the diagonal entry (pc, pc)
//
// The interleaving is load-bearing. An earlier version used stride slip_D for columns
// and appended the padding as a contiguous zero band [slip_D*Np, D*Np). That gave rows
// and columns different strides, so row i and column i referred to *different nodes* and
// the square matrix's diagonal was not the physical near-field self-interaction. At one
// rank the tree never subdivides and nothing depends on that alignment, but as soon as
// it splits, every diagonal block is a misaligned far-field block — errors of 1e5 and
// worse, growing with tree depth. Interleaving restores row/column alignment and, as a
// side effect, makes an all-zero leaf impossible. The scaled unit diagonal keeps each
// fictitious column nonzero on its own. apply() feeds zeros into these columns, so
// their content never reaches the output.
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

    // Diagonal value for the fictitious columns: the RMS 2-norm of a real column of G.
    // Dimensionally consistent with the columns it stands in for, so the padded matrix
    // is neither near-singular (which is what blew up) nor dominated by the padding.
    {
        PetscReal fro;
        CHKERRTHROW(MatNorm(G_dense, NORM_FROBENIUS, &fro));
        pad_scale_ = (N_real > 0) ? static_cast<double>(fro) / std::sqrt(static_cast<double>(N_real))
                                  : 1.0;
        if (pad_scale_ <= 0.0 || !std::isfinite(pad_scale_)) pad_scale_ = 1.0;
    }

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

    // HODLRMatrix's matrix-free compress() callback does not receive the row/column
    // distributions the way structured::construct_matrix_free did, so we capture them from
    // the matrix itself. The padded operator is square and rows and columns share one
    // ClusterTree, hence a single distribution serves as both rdist and cdist. Filled in
    // from H->dist() before compress() is called.
    std::vector<int> hodlr_dist;

    auto Amult = [&mvc_ref, &adc_ref,
                  &cb_fwd, &cb_adj,
                  &x_full,
                  &x_tandem_slip, &x_tandem_trac,
                  &y_full_row, &y_full_col,
                  &y_local_row, &y_local_col,
                  &row_counts, &row_displs,
                  &col_counts, &col_displs,
                  &hodlr_dist,
                  this, G_dense, n_ranks, my_rank, comm_cap]
        (Trans t,
         const DenseMatrix<double>& R,
         DenseMatrix<double>& S) {

        const std::vector<int>& rdist = hodlr_dist;
        const std::vector<int>& cdist = hodlr_dist;

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

        // The allgatherv below reads in_counts[my_rank] contiguous doubles from column k
        // of R, and step 5 writes strumpack_out_local entries into column k of S. Both
        // assume STRUMPACK's local block sizes agree with the rdist/cdist deltas it just
        // handed us. apply() distrusts this and re-queries rdist()/cdist(); assert it here
        // so a layout mismatch fails loudly instead of silently corrupting the sample.
        assert(static_cast<int>(R.rows()) == in_counts[my_rank]);
        assert(static_cast<int>(S.rows()) == strumpack_out_local);
        assert(R.ld() >= R.rows() && S.ld() >= S.rows());

        for (int k = 0; k < nvec; ++k) {
            // Step 1: allgather STRUMPACK input → x_full (D*Np, permuted order)
            CHKERRTHROW(MPI_Allgatherv(
                R.ptr(0, k), in_counts[my_rank], MPI_DOUBLE,
                x_full.data(), in_counts.data(), in_displs.data(), MPI_DOUBLE, comm_cap));

            if (fwd) {
                // Step 2 (fwd): scatter the real slip slots (map >= 0) into Tandem order.
                //               Fictitious slots are skipped here and handled in step 5.
                for (PetscInt pc = 0; pc < static_cast<PetscInt>(x_full.size()); ++pc) {
                    const PetscInt tc = this->col_perm_to_tandem_[pc];
                    if (tc >= 0) x_tandem_slip[tc] = x_full[pc];
                }

                // Step 3 (fwd): PETSc distributed MatMult  y = G_dense * slip.
                // x_tandem_slip holds the full global slip (tandem order) on every rank;
                // col_work_ wraps this rank's local column slice, row_work_ wraps the local
                // row output. PETSc owns the parallel matvec (correct for any rank count),
                // replacing a hand-rolled local dense loop that assumed a single-rank layout.
                CHKERRTHROW(VecPlaceArray(this->col_work_,
                                          x_tandem_slip.data() + this->petsc_dist_col_[my_rank]));
                CHKERRTHROW(VecPlaceArray(this->row_work_, y_local_row.data()));
                CHKERRTHROW(MatMult(G_dense, this->col_work_, this->row_work_));
                CHKERRTHROW(VecResetArray(this->col_work_));
                CHKERRTHROW(VecResetArray(this->row_work_));

                // Step 4 (fwd): allgather MatMult output (D*Np, Tandem order) → y_full_row
                CHKERRTHROW(MPI_Allgatherv(
                    y_local_row.data(), row_counts[my_rank], MPI_DOUBLE,
                    y_full_row.data(), row_counts.data(), row_displs.data(), MPI_DOUBLE, comm_cap));

                // Step 5 (fwd): reorder Tandem → permuted row, write STRUMPACK output slice.
                // Rows and columns share a layout, so the fictitious column at index pr
                // contributes pad_scale_ * x[pr] to row pr — a pure diagonal term.
                for (int i = 0; i < strumpack_out_local; ++i) {
                    const int pr = strumpack_out_start + i;
                    double v = y_full_row[this->row_perm_to_tandem_[pr]];
                    if (this->is_pad_index(pr)) v += this->pad_scale_ * x_full[pr];
                    S.ptr(0, k)[i] = v;
                }

            } else {
                // Step 2 (adj): reorder full row input → x_tandem_trac
                for (PetscInt pr = 0; pr < static_cast<PetscInt>(x_full.size()); ++pr)
                    x_tandem_trac[row_perm_to_tandem_[pr]] = x_full[pr];

                // Step 3 (adj): PETSc distributed transpose matvec  y = G_dense^T * trac.
                // x_tandem_trac holds the full global traction (tandem order) on every rank;
                // row_work_ wraps this rank's local row slice, col_work_ wraps the local
                // column output. PETSc owns the parallel transpose matvec.
                CHKERRTHROW(VecPlaceArray(this->row_work_,
                                          x_tandem_trac.data() + this->petsc_dist_row_[my_rank]));
                CHKERRTHROW(VecPlaceArray(this->col_work_, y_local_col.data()));
                CHKERRTHROW(MatMultTranspose(G_dense, this->row_work_, this->col_work_));
                CHKERRTHROW(VecResetArray(this->row_work_));
                CHKERRTHROW(VecResetArray(this->col_work_));

                // Step 4 (adj): gather PETSc's distributed column output → full tandem-order y_full_col
                CHKERRTHROW(MPI_Allgatherv(
                    y_local_col.data(), col_counts[my_rank], MPI_DOUBLE,
                    y_full_col.data(), col_counts.data(), col_displs.data(), MPI_DOUBLE, comm_cap));

                // Step 5 (adj): reorder permuted col → Tandem, write STRUMPACK output slice.
                // A fictitious column has only its diagonal entry, so the transpose
                // contributes pad_scale_ * R[po] there and nothing from G.
                for (int i = 0; i < strumpack_out_local; ++i) {
                    const int po = strumpack_out_start + i;
                    const PetscInt tc = this->col_perm_to_tandem_[po];
                    S.ptr(0, k)[i] = (tc >= 0)
                                     ? y_full_col[tc]
                                     : this->pad_scale_ * x_full[po];
                }
            }
        }
    };

    // Cluster tree controlling STRUMPACK's hierarchical blocking. Rows and columns share
    // the node-interleaved layout, so a single tree over the D*Np space is valid for both.
    //   "kdtree"  — spatial median-split tree recursing to leaf_size; depth is a function
    //               of leaf_size alone, so the compression is RANK-INDEPENDENT.
    //   "petsc1d" — binary bisection over ranks. Kept reachable for reproducing older
    //               runs, but note its depth is log2(n_ranks): adding processes changes
    //               the hierarchy, hence the compression and the error. Not comparable
    //               across rank counts.
    const bool use_kdtree = (config_.cluster_tree == "kdtree");
    ClusterTree row_tree = use_kdtree ? *kd_tree_
                                      : build_petsc_tree(petsc_dist_row_, 0, n_ranks);

    // Construct HODLR::HODLRMatrix directly rather than going through
    // structured::construct_matrix_free. That wrapper converts StructuredOptions ->
    // HODLROptions with a constructor that copies only the base-class fields, so every
    // ButterflyPACK knob (less_adapt, rank_guess, rank_rate, sample_para, butterfly levels)
    // silently reverts to its default and is unreachable from config. HODLRMatrix derives
    // from structured::StructuredMatrix, so S_full_, apply(), mult() and memory() are
    // unaffected by the switch.
    HODLR::HODLROptions<double> opts;
    opts.set_rel_tol(config_.rtol);
    opts.set_leaf_size(config_.leaf_size);
    opts.set_less_adapt(config_.less_adapt);
    opts.set_rank_guess(config_.rank_guess);
    opts.set_rank_rate(config_.rank_rate);
    opts.set_BF_sampling_parameter(config_.bf_sampling);
    // "hodbf" = butterfly off-diagonal blocks. STRUMPACK's own HODBF path sets these two.
    const bool use_hodbf = (config_.format == "hodbf");
    if (use_hodbf) {
        opts.set_butterfly_levels(1000);
        opts.set_BF_entry_n15(true);
    }
    // NOTE: max_rank is deliberately not set — HODLRMatrix never reads it (see HMatrixConfig).

    if (my_rank == 0) {
        std::cout << "  Building S_full  (" << M_tot << " x " << N_tot
                  << " padded-square, " << N_real << " real cols"
                  << ", node-interleaved, pad_scale=" << pad_scale_ << ")\n"
                  << "  opts[" << (use_hodbf ? "hodbf" : "hodlr") << "]: rtol=" << config_.rtol
                  << " leaf_size=" << config_.leaf_size
                  << " less_adapt=" << (config_.less_adapt ? "true" : "false")
                  << " rank_guess=" << config_.rank_guess
                  << " rank_rate=" << config_.rank_rate
                  << " bf_sampling=" << config_.bf_sampling << "\n";
        report_tree(row_tree, D_, slip_D_, use_kdtree ? "kdtree" : "petsc1d", n_ranks);
    }

    {
        // Two-step: build the (empty) tree-defined matrix so its row distribution can be
        // read, publish it to the callback, then compress.
        auto H = std::make_unique<HODLR::HODLRMatrix<double>>(MPIComm(comm_), row_tree, opts);
        hodlr_dist = H->dist();
        if (static_cast<int>(hodlr_dist.size()) != n_ranks + 1)
            throw std::runtime_error("HODLRMatrix::dist() has unexpected length");
        if (hodlr_dist.back() != static_cast<int>(M_tot))
            throw std::runtime_error("HODLRMatrix::dist() does not span the padded row space");
        H->compress(Amult);
        S_full_ = std::move(H);
    }

    // StructuredMatrix::memory() is this rank's share only (HODLRMatrix pairs it with a
    // total_memory() that all-reduces, but that is not on the base class). Reducing here
    // is required for the reported figure to be comparable against the global dense size
    // — without it the "compression ratio" silently improves by a factor of n_ranks.
    {
        double local_mem = static_cast<double>(S_full_->memory());
        double total_mem = 0.0;
        CHKERRTHROW(MPI_Allreduce(&local_mem, &total_mem, 1, MPI_DOUBLE, MPI_SUM, comm_));
        mem_bytes_ = total_mem;
    }
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
        // s_perm_work_ lives in the COMPACT permuted slip space (slip_D*Np, stride slip_D),
        // which is what PETSc distributes; col_perm_to_tandem_ is indexed in the PADDED
        // space (stride D). Map compact -> padded to look up the Tandem column.
        std::vector<PetscInt> from_idx(local_col);
        for (PetscInt pc = perm_cstart; pc < perm_cend; ++pc) {
            const PetscInt padded = (pc / slip_D_) * D_ + (pc % slip_D_);
            from_idx[pc - perm_cstart] = col_perm_to_tandem_[padded];
        }
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

    // Gather s_perm in the compact permuted slip space (slip_D*Np, stride slip_D), then
    // expand into the padded D*Np input vector (stride D). Fictitious slots stay zero,
    // which is why the padding's content never reaches the output.
    const PetscInt N_real = slip_D_ * Np;
    std::vector<double> s_perm_full(N_real);
    {
        const PetscScalar* s_arr;
        CHKERRTHROW(VecGetArrayRead(s_perm_work_, &s_arr));
        CHKERRTHROW(MPI_Allgatherv(s_arr, col_counts[my_rank], MPI_DOUBLE,
                                   s_perm_full.data(), col_counts.data(),
                                   col_displs.data(), MPI_DOUBLE, comm_));
        CHKERRTHROW(VecRestoreArrayRead(s_perm_work_, &s_arr));
    }

    std::vector<double> x_perm_padded(D_ * Np, 0.0);  // zero-initialise covers padding
    for (PetscInt new_j = 0; new_j < Np; ++new_j)
        for (int beta = 0; beta < slip_D_; ++beta)
            x_perm_padded[new_j * D_ + beta] = s_perm_full[new_j * slip_D_ + beta];

    // Step 3: extract STRUMPACK's local input slice from the padded vector, call mult
    const auto& cdist = S_full_->cdist();
    const auto& rdist = S_full_->rdist();
    const int s_in_local  = cdist[my_rank+1] - cdist[my_rank];
    const int s_out_local = rdist[my_rank+1] - rdist[my_rank];

    strumpack::DenseMatrix<double> x_dm(s_in_local, 1);
    std::memcpy(x_dm.data(), x_perm_padded.data() + cdist[my_rank], s_in_local * sizeof(double));

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
