#include "StrumpackGFOperator.h"
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
// PCA helpers — identical to HMatrixGreenFunction (detects fault dimensionality)
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

StrumpackGFOperator::StrumpackGFOperator(
    Mat G_dense,
    const std::vector<PetscReal>& local_coords,
    PetscInt nbf, int D,
    Vec s_proto, Vec t_proto,
    MPI_Comm comm,
    HMatrixConfig const& config,
    strumpack::structured::Type format)
    : D_(D), slip_D_(D - 1), nbf_(nbf), comm_(comm), config_(config), format_(format)
{
    static_assert(sizeof(PetscScalar) == sizeof(double),
                  "StrumpackGFOperator requires real (non-complex) PETSc scalar");

    int n_ranks, my_rank;
    CHKERRTHROW(MPI_Comm_size(comm_, &n_ranks));
    CHKERRTHROW(MPI_Comm_rank(comm_, &my_rank));

    PetscInt M_global, N_global;
    CHKERRTHROW(MatGetSize(G_dense, &M_global, &N_global));
    N_el_ = M_global / (D_ * nbf_);
    assert(N_el_ * D_      * nbf_ == M_global);
    assert(N_el_ * slip_D_ * nbf_ == N_global);

    // Precompute PETSc's PETSC_DECIDE row/col distribution for the Np×Np sub-matrices.
    // This matches MatCreateDense(..., PETSC_DECIDE, PETSC_DECIDE, Np, Np, ...).
    {
        const PetscInt Np = N_el_ * nbf_;
        petsc_dist_.resize(n_ranks + 1, 0);
        for (int r = 0; r < n_ranks; ++r)
            petsc_dist_[r+1] = petsc_dist_[r] + Np/n_ranks + (r < Np%n_ranks ? 1 : 0);
    }

    // ---- Step 1: Allgather local coords → global (N_el*nbf × D) ----
    PetscInt local_n = static_cast<PetscInt>(local_coords.size() / D_);
    std::vector<PetscInt> all_n(n_ranks);
    CHKERRTHROW(MPI_Allgather(&local_n, 1, MPIU_INT, all_n.data(), 1, MPIU_INT, comm_));
    std::vector<int> rcounts(n_ranks), displs(n_ranks, 0);
    for (int r = 0; r < n_ranks; ++r) rcounts[r] = static_cast<int>(all_n[r]) * D_;
    for (int r = 1; r < n_ranks; ++r) displs[r] = displs[r-1] + rcounts[r-1];
    std::vector<PetscReal> global_coords(N_el_ * nbf_ * D_);
    CHKERRTHROW(MPI_Allgatherv(local_coords.data(),
                               static_cast<int>(local_coords.size()), MPIU_REAL,
                               global_coords.data(), rcounts.data(), displs.data(),
                               MPIU_REAL, comm_));

    // ---- Step 2: Spatial permutation (PCA-Hilbert on N_el*nbf scalar cloud) ----
    build_spatial_permutation(global_coords);

    // ---- Step 3: Build D*(D-1) STRUMPACK matrices ----
    const int n_ab = D_ * slip_D_;
    S_ab_.resize(n_ab);
    mem_bytes_ab_.resize(n_ab, 0.0);

    // planar_fault check: verify and skip normal-traction (α=0) components.
    if (config_.planar_fault) {
        PetscReal ref_norm = 0.0;
        std::vector<PetscReal> norms_ab(n_ab, 0.0);
        for (int alpha = 0; alpha < D_; ++alpha)
            for (int beta = 0; beta < slip_D_; ++beta) {
                Mat G_ab = extract_component_submatrix(G_dense, alpha, beta);
                CHKERRTHROW(MatNorm(G_ab, NORM_FROBENIUS, &norms_ab[alpha * slip_D_ + beta]));
                CHKERRTHROW(MatDestroy(&G_ab));
                if (alpha > 0) ref_norm = std::max(ref_norm, norms_ab[alpha * slip_D_ + beta]);
            }
        if (my_rank == 0 && ref_norm > 0.0)
            for (int beta = 0; beta < slip_D_; ++beta)
                std::cout << "  planar_fault check: ||G_0" << beta
                          << "|| / ||G_shear_max|| = "
                          << norms_ab[beta] / static_cast<double>(ref_norm) << "\n";

        constexpr double TOL = 1e-6;
        for (int beta = 0; beta < slip_D_; ++beta) {
            double ratio = (ref_norm > 0.0)
                           ? norms_ab[beta] / static_cast<double>(ref_norm) : 0.0;
            if (ratio > TOL) {
                if (my_rank == 0)
                    std::cerr << "\nERROR: planar_fault=true but G_0" << beta
                              << " is not negligible (ratio=" << ratio << ")\n";
                throw std::runtime_error(
                    "planar_fault=true validation failed: normal-traction GF not negligible");
            }
        }
    }

    for (int alpha = 0; alpha < D_; ++alpha) {
        for (int beta = 0; beta < slip_D_; ++beta) {
            if (config_.planar_fault && alpha == 0) {
                if (my_rank == 0)
                    std::cout << "  Skipping S_0" << beta
                              << "  (normal traction, zero by planar-fault symmetry)\n";
                continue;
            }
            if (my_rank == 0)
                std::cout << "  Building S_" << alpha << beta
                          << "  (traction comp " << alpha << " <- slip comp " << beta << ")\n";

            Mat G_ab   = extract_component_submatrix(G_dense, alpha, beta);
            Mat G_perm = build_spatially_permuted_submatrix(G_ab);
            CHKERRTHROW(MatDestroy(&G_ab));
            build_one_s_matrix(alpha, beta, G_perm);
            CHKERRTHROW(MatDestroy(&G_perm));
        }
    }

    // ---- Step 4: Per-component scatter objects ----
    build_scatters(s_proto, t_proto);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

StrumpackGFOperator::~StrumpackGFOperator() {
    for (auto& v : slip_spatial_)     VecDestroy(&v);
    for (auto& v : traction_spatial_) VecDestroy(&v);
    for (auto& s : scatter_s_)        VecScatterDestroy(&s);
    for (auto& s : scatter_t_)        VecScatterDestroy(&s);
}

// ---------------------------------------------------------------------------
// build_spatial_permutation — PCA-Hilbert ordering (identical to HMatrixGreenFunction)
// ---------------------------------------------------------------------------

void StrumpackGFOperator::build_spatial_permutation(
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
    // level, down to leaf_size). Store it once for all S_αβ. For eff_dim==1 the median split
    // degenerates to the same sorted order as the Hilbert branch below.
    if (config_.cluster_tree == "kdtree") {
        std::vector<PetscInt> indices(Np);
        std::iota(indices.begin(), indices.end(), PetscInt{0});
        PetscInt fill = 0;
        kd_tree_ = std::make_unique<strumpack::structured::ClusterTree>(
            build_kdtree(indices, proj, eff_dim, config_.leaf_size, fill));
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
// extract_component_submatrix — identical to HMatrixGreenFunction
// ---------------------------------------------------------------------------

Mat StrumpackGFOperator::extract_component_submatrix(Mat G_dense, int alpha, int beta) const {
    PetscInt Np        = N_el_ * nbf_;
    PetscInt row_block = D_     * nbf_;
    PetscInt col_block = slip_D_ * nbf_;

    std::vector<PetscInt> src_cols(Np), dst_cols(Np);
    for (PetscInt e = 0; e < N_el_; ++e)
        for (PetscInt n = 0; n < nbf_; ++n) {
            PetscInt j  = e * nbf_ + n;
            src_cols[j] = e * col_block + beta * nbf_ + n;
            dst_cols[j] = j;
        }

    PetscInt gf_rstart, gf_rend;
    CHKERRTHROW(MatGetOwnershipRange(G_dense, &gf_rstart, &gf_rend));
    const PetscInt first_elem = gf_rstart / row_block;
    const PetscInt last_elem  = gf_rend   / row_block;

    Mat G_ab;
    CHKERRTHROW(MatCreateDense(comm_, PETSC_DECIDE, PETSC_DECIDE, Np, Np, nullptr, &G_ab));

    std::vector<PetscInt>    src_rows(nbf_), dst_rows(nbf_);
    std::vector<PetscScalar> block(nbf_ * Np);

    for (PetscInt e = first_elem; e < last_elem; ++e) {
        for (PetscInt n = 0; n < nbf_; ++n) {
            src_rows[n] = e * row_block + alpha * nbf_ + n;
            dst_rows[n] = e * nbf_ + n;
        }
        CHKERRTHROW(MatGetValues(G_dense, nbf_, src_rows.data(), Np, src_cols.data(), block.data()));
        CHKERRTHROW(MatSetValues(G_ab, nbf_, dst_rows.data(), Np, dst_cols.data(), block.data(),
                                 INSERT_VALUES));
    }
    CHKERRTHROW(MatAssemblyBegin(G_ab, MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd  (G_ab, MAT_FINAL_ASSEMBLY));
    return G_ab;
}

// ---------------------------------------------------------------------------
// build_spatially_permuted_submatrix — identical to HMatrixGreenFunction
// ---------------------------------------------------------------------------

Mat StrumpackGFOperator::build_spatially_permuted_submatrix(Mat G_ab) const {
    PetscInt Np = N_el_ * nbf_;

    std::vector<PetscInt> inv_perm(Np);
    for (PetscInt ni = 0; ni < Np; ++ni) inv_perm[perm_[ni]] = ni;

    PetscInt old_rstart, old_rend;
    CHKERRTHROW(MatGetOwnershipRange(G_ab, &old_rstart, &old_rend));

    Mat G_perm;
    CHKERRTHROW(MatCreateDense(comm_, PETSC_DECIDE, PETSC_DECIDE, Np, Np, nullptr, &G_perm));

    std::vector<PetscInt>    all_old_cols(Np), all_new_cols(Np);
    std::iota(all_old_cols.begin(), all_old_cols.end(), 0);
    std::iota(all_new_cols.begin(), all_new_cols.end(), 0);
    std::vector<PetscScalar> row_old(Np), row_new(Np);

    for (PetscInt old_i = old_rstart; old_i < old_rend; ++old_i) {
        CHKERRTHROW(MatGetValues(G_ab, 1, &old_i, Np, all_old_cols.data(), row_old.data()));
        const PetscInt new_i = inv_perm[old_i];
        for (PetscInt old_j = 0; old_j < Np; ++old_j)
            row_new[inv_perm[old_j]] = row_old[old_j];
        CHKERRTHROW(MatSetValues(G_perm, 1, &new_i, Np, all_new_cols.data(), row_new.data(),
                                 INSERT_VALUES));
    }
    CHKERRTHROW(MatAssemblyBegin(G_perm, MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd  (G_perm, MAT_FINAL_ASSEMBLY));
    return G_perm;
}

// ---------------------------------------------------------------------------
// build_petsc_tree — binary ClusterTree matching PETSc PETSC_DECIDE distribution
//
// dist[r] = first row of rank r, dist[n_ranks] = Np.
// The resulting tree guarantees that rdist[p] == dist[p] in every mult_1d_t
// callback, enabling zero-copy VecPlaceArray without any redistribution.
// ---------------------------------------------------------------------------

strumpack::structured::ClusterTree
StrumpackGFOperator::build_petsc_tree(const std::vector<int>& dist, int lo, int hi) {
    strumpack::structured::ClusterTree t(dist[hi] - dist[lo]);
    if (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        t.c.push_back(build_petsc_tree(dist, lo, mid));
        t.c.push_back(build_petsc_tree(dist, mid, hi));
    }
    return t;
}

// ---------------------------------------------------------------------------
// build_kdtree — spatial median-split ClusterTree that INDUCES perm_
//
// Each leaf writes its indices contiguously into perm_ (left subtree entirely before
// right), so every node covers a contiguous perm_ range AND a spatially compact tile —
// tree↔perm_ consistency by construction. Recurses to leaf_size, not n_ranks, so the tree
// is spatially separated at every level ButterflyPACK can subdivide.
// ---------------------------------------------------------------------------

strumpack::structured::ClusterTree
StrumpackGFOperator::build_kdtree(std::vector<PetscInt>& indices,
                                  const std::vector<double>& proj, int eff_dim,
                                  int leaf_size, PetscInt& fill) {
    const std::size_t n = indices.size();
    strumpack::structured::ClusterTree node(static_cast<int>(n));

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
// build_one_s_matrix — STRUMPACK BUTTERFLY matrix via zero-copy mult_1d_t
// ---------------------------------------------------------------------------

void StrumpackGFOperator::build_one_s_matrix(int alpha, int beta, Mat G_perm_ab) {
    using namespace strumpack;
    using namespace strumpack::structured;

    int n_ranks, my_rank;
    CHKERRTHROW(MPI_Comm_size(comm_, &n_ranks));
    CHKERRTHROW(MPI_Comm_rank(comm_, &my_rank));

    const PetscInt Np = N_el_ * nbf_;

    // PETSc local row/col ownership from petsc_dist_ (precomputed in constructor).
    const PetscInt perm_rstart = static_cast<PetscInt>(petsc_dist_[my_rank]);
    const PetscInt perm_rend   = static_cast<PetscInt>(petsc_dist_[my_rank + 1]);
    const PetscInt perm_cstart = perm_rstart;   // square, same distribution
    const PetscInt perm_cend   = perm_rend;
    const PetscInt local_m = perm_rend - perm_rstart;
    const PetscInt local_n = local_m;

    // Verify G_perm_ab ownership matches our precomputed distribution.
    {
        PetscInt actual_rstart, actual_rend;
        CHKERRTHROW(MatGetOwnershipRange(G_perm_ab, &actual_rstart, &actual_rend));
        assert(actual_rstart == perm_rstart && actual_rend == perm_rend);
    }

    // Cluster tree controlling STRUMPACK's hierarchical blocking. "kdtree": spatial median-split
    // tree (built once in the ctor, shared by all S_αβ) so off-diagonal blocks stay spatially
    // separated at every level. "petsc1d": binary bisection matching PETSc's layout.
    // Either way, ButterflyPACK may use its own internal 1D distribution that differs from this
    // tree's leaf layout; the callback below handles any distribution via allgatherv.
    ClusterTree row_tree = (config_.cluster_tree == "kdtree")
        ? *kd_tree_
        : build_petsc_tree(petsc_dist_, 0, n_ranks);

    // Pre-allocate PETSc work Vecs (sized to PETSc's layout, reused for all callback calls).
    Vec x_work, y_work;
    CHKERRTHROW(VecCreateMPIWithArray(comm_, 1, local_n, Np, nullptr, &x_work));
    CHKERRTHROW(VecCreateMPIWithArray(comm_, 1, local_m, Np, nullptr, &y_work));

    // Precompute PETSc distribution counts for allgatherv.
    std::vector<int> petsc_counts(n_ranks), petsc_displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) petsc_counts[r] = petsc_dist_[r+1] - petsc_dist_[r];
    petsc_displs[0] = 0;
    for (int r = 1; r < n_ranks; ++r) petsc_displs[r] = petsc_dist_[r];

    // Per-callback scratch buffers (allocated once, reused across all callback calls).
    std::vector<double> x_full(Np), y_full(Np), y_local(local_m);

    auto& mvc_ref = matvec_count_;
    auto& adc_ref = adjoint_count_;
    MPI_Comm comm_cap = comm_;
    std::size_t cb_fwd = 0, cb_adj = 0;  // callback-invocation counters (per component)

    // mult_1d_t callback — called collectively by all ranks.
    //
    // DISTRIBUTION NOTE: ButterflyPACK computes its own internal 1D distribution that
    // may differ from our cluster tree's leaf layout (e.g. N_leaf=43 vs 140 per rank).
    // Consequently rdist/cdist in the callback are ButterflyPACK's layout, NOT PETSc's.
    //
    // We bridge via allgatherv:
    //   1. Allgather R[:,k] from STRUMPACK's input distribution → x_full (global, all ranks)
    //   2. MatMult(G_perm_ab, x_petsc, y_petsc) using PETSc's local slice of x_full
    //   3. Allgather y_local from PETSc's output distribution → y_full
    //   4. Copy y_full[rdist[p]..rdist[p+1]] → S[:,k] (STRUMPACK's expected output slice)
    //
    // Cost: O(2 * Np * nvec) communication per callback call — fine for Np≤1e5.
    // For Np~1e6 production, replace with the zero-copy VecPlaceArray approach once
    // we confirm distributions match (requires STRUMPACK/ButterflyPACK distribution audit).
    auto Amult = [&mvc_ref, &adc_ref,
                  &cb_fwd, &cb_adj,
                  &x_work, &y_work,
                  &x_full, &y_full, &y_local,
                  &petsc_counts, &petsc_displs,
                  perm_rstart, perm_cstart, local_m, local_n,
                  G_perm_ab, n_ranks, my_rank, Np, comm_cap,
                  alpha, beta]
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
            std::cout << "    [S_" << alpha << beta
                      << " " << (t == Trans::N ? "Gv  " : "G^Tw")
                      << " #" << cb << "] nvec=" << nvec
                      << "  running fwd=" << mvc_ref.load()
                      << " adj=" << adc_ref.load() << "\n" << std::flush;

        // STRUMPACK's input/output distributions for this call
        const std::vector<int>& in_dist  = (t == Trans::N) ? cdist : rdist;
        const std::vector<int>& out_dist = (t == Trans::N) ? rdist : cdist;

        // Counts for allgatherv of the input (STRUMPACK's distribution)
        std::vector<int> in_counts(n_ranks), in_displs(n_ranks);
        for (int r = 0; r < n_ranks; ++r) in_counts[r] = in_dist[r+1] - in_dist[r];
        in_displs[0] = 0;
        for (int r = 1; r < n_ranks; ++r) in_displs[r] = in_dist[r];

        // PETSc input/output offsets
        const PetscInt petsc_in_start  = (t == Trans::N) ? perm_cstart : perm_rstart;
        const PetscInt petsc_in_local  = (t == Trans::N) ? local_n     : local_m;
        const PetscInt petsc_out_start = (t == Trans::N) ? perm_rstart : perm_cstart;
        const PetscInt petsc_out_local = (t == Trans::N) ? local_m     : local_n;

        // STRUMPACK's output slice on this rank
        const int strumpack_out_start = out_dist[my_rank];
        const int strumpack_out_local = out_dist[my_rank+1] - out_dist[my_rank];

        for (int k = 0; k < nvec; ++k) {
            // Step 1: gather full input vector from STRUMPACK's distribution
            CHKERRTHROW(MPI_Allgatherv(
                R.ptr(0, k), in_counts[my_rank], MPI_DOUBLE,
                x_full.data(), in_counts.data(), in_displs.data(), MPI_DOUBLE, comm_cap));

            // Step 2: PETSc MatMult — use our local slice of x_full
            CHKERRTHROW(VecPlaceArray(x_work, x_full.data() + petsc_in_start));
            CHKERRTHROW(VecPlaceArray(y_work, y_local.data()));
            if (t == Trans::N)
                CHKERRTHROW(MatMult         (G_perm_ab, x_work, y_work));
            else
                CHKERRTHROW(MatMultTranspose(G_perm_ab, x_work, y_work));
            CHKERRTHROW(VecResetArray(x_work));
            CHKERRTHROW(VecResetArray(y_work));

            // Step 3: gather PETSc output from all ranks
            CHKERRTHROW(MPI_Allgatherv(
                y_local.data(), petsc_counts[my_rank], MPI_DOUBLE,
                y_full.data(), petsc_counts.data(), petsc_displs.data(), MPI_DOUBLE, comm_cap));

            // Step 4: extract STRUMPACK's expected output slice
            std::memcpy(S.ptr(0, k),
                        y_full.data() + strumpack_out_start,
                        strumpack_out_local * sizeof(double));
        }
    };

    StructuredOptions<double> opts(format_);
    opts.set_rel_tol(config_.rtol);
    opts.set_leaf_size(config_.leaf_size);
    opts.set_max_rank(config_.max_rank);

    const int idx = alpha * slip_D_ + beta;
    S_ab_[idx] = construct_matrix_free<double>(
        MPIComm(comm_), static_cast<int>(Np), static_cast<int>(Np),
        Amult, opts, &row_tree);

    mem_bytes_ab_[idx] = static_cast<double>(S_ab_[idx]->memory());

    CHKERRTHROW(VecDestroy(&x_work));
    CHKERRTHROW(VecDestroy(&y_work));
}

// ---------------------------------------------------------------------------
// build_scatters — create per-component VecScatter objects
// Uses STRUMPACK's dist() to determine local row/col sizes, which match
// PETSc's layout by construction (same tree).
// ---------------------------------------------------------------------------

void StrumpackGFOperator::build_scatters(Vec s_proto, Vec t_proto) {
    const PetscInt Np = N_el_ * nbf_;

    bool any_built = false;
    for (auto& S : S_ab_) { if (S) { any_built = true; break; } }
    if (!any_built)
        throw std::runtime_error("StrumpackGFOperator: all S_ab_ are null — nothing to scatter");

    int my_rank;
    CHKERRTHROW(MPI_Comm_rank(comm_, &my_rank));

    // Use PETSc's distribution (petsc_dist_) for the scatter/work-vec infrastructure.
    // This is the same layout used inside the mult callback.
    const PetscInt perm_rstart = static_cast<PetscInt>(petsc_dist_[my_rank]);
    const PetscInt perm_rend   = static_cast<PetscInt>(petsc_dist_[my_rank + 1]);
    const PetscInt perm_cstart = perm_rstart;
    const PetscInt perm_cend   = perm_rend;
    const PetscInt local_m = perm_rend - perm_rstart;
    const PetscInt local_n = local_m;

    slip_spatial_.resize(slip_D_, nullptr);
    traction_spatial_.resize(D_, nullptr);
    for (int beta  = 0; beta  < slip_D_; ++beta)
        CHKERRTHROW(VecCreateMPI(comm_, local_n, Np, &slip_spatial_[beta]));
    for (int alpha = 0; alpha < D_;      ++alpha)
        CHKERRTHROW(VecCreateMPI(comm_, local_m, Np, &traction_spatial_[alpha]));

    // Slip scatters: s β-component slice (original) → slip_spatial_[β]
    scatter_s_.resize(slip_D_, nullptr);
    {
        std::vector<PetscInt> from_idx(local_n);
        for (int beta = 0; beta < slip_D_; ++beta) {
            for (PetscInt nj = perm_cstart; nj < perm_cend; ++nj) {
                PetscInt scalar = perm_[nj];
                PetscInt e = scalar / nbf_, n = scalar % nbf_;
                from_idx[nj - perm_cstart] = e * slip_D_ * nbf_ + beta * nbf_ + n;
            }
            IS from_IS, to_IS;
            CHKERRTHROW(ISCreateGeneral(comm_, local_n, from_idx.data(),
                                        PETSC_COPY_VALUES, &from_IS));
            CHKERRTHROW(ISCreateStride (comm_, local_n, perm_cstart, 1, &to_IS));
            CHKERRTHROW(VecScatterCreate(s_proto, from_IS,
                                         slip_spatial_[beta], to_IS, &scatter_s_[beta]));
            CHKERRTHROW(ISDestroy(&from_IS));
            CHKERRTHROW(ISDestroy(&to_IS));
        }
    }

    // Traction scatters: traction_spatial_[α] → t α-component slice (original)
    scatter_t_.resize(D_, nullptr);
    {
        std::vector<PetscInt> to_idx(local_m);
        for (int alpha = 0; alpha < D_; ++alpha) {
            for (PetscInt ni = perm_rstart; ni < perm_rend; ++ni) {
                PetscInt scalar = perm_[ni];
                PetscInt e = scalar / nbf_, n = scalar % nbf_;
                to_idx[ni - perm_rstart] = e * D_ * nbf_ + alpha * nbf_ + n;
            }
            IS from_IS, to_IS;
            CHKERRTHROW(ISCreateStride (comm_, local_m, perm_rstart, 1, &from_IS));
            CHKERRTHROW(ISCreateGeneral(comm_, local_m, to_idx.data(),
                                        PETSC_COPY_VALUES, &to_IS));
            CHKERRTHROW(VecScatterCreate(traction_spatial_[alpha], from_IS,
                                         t_proto, to_IS, &scatter_t_[alpha]));
            CHKERRTHROW(ISDestroy(&from_IS));
            CHKERRTHROW(ISDestroy(&to_IS));
        }
    }
}

// ---------------------------------------------------------------------------
// apply — t += Σ_{α,β} S_αβ * s_β   (both in original Tandem DOF order)
// ---------------------------------------------------------------------------

void StrumpackGFOperator::apply(Vec s, Vec t) const {
    const PetscInt Np = N_el_ * nbf_;

    int n_ranks, my_rank;
    CHKERRTHROW(MPI_Comm_size(comm_, &n_ranks));
    CHKERRTHROW(MPI_Comm_rank(comm_, &my_rank));

    // PETSc distribution counts (for allgatherv of slip/traction spatial vecs)
    std::vector<int> petsc_counts(n_ranks), petsc_displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) petsc_counts[r] = petsc_dist_[r+1] - petsc_dist_[r];
    petsc_displs[0] = 0;
    for (int r = 1; r < n_ranks; ++r) petsc_displs[r] = petsc_dist_[r];

    const PetscInt petsc_local = static_cast<PetscInt>(petsc_dist_[my_rank+1] - petsc_dist_[my_rank]);

    // Scatter all slip components to spatial order (PETSc's layout)
    for (int beta = 0; beta < slip_D_; ++beta) {
        CHKERRTHROW(VecScatterBegin(scatter_s_[beta], s, slip_spatial_[beta],
                                    INSERT_VALUES, SCATTER_FORWARD));
        CHKERRTHROW(VecScatterEnd  (scatter_s_[beta], s, slip_spatial_[beta],
                                    INSERT_VALUES, SCATTER_FORWARD));
    }

    std::vector<double> x_full(Np), y_full(Np);

    for (int alpha = 0; alpha < D_; ++alpha) {
        CHKERRTHROW(VecZeroEntries(traction_spatial_[alpha]));
        bool any_nonzero = false;

        PetscScalar *t_arr;
        CHKERRTHROW(VecGetArray(traction_spatial_[alpha], &t_arr));

        for (int beta = 0; beta < slip_D_; ++beta) {
            auto& Sab = S_ab_[alpha * slip_D_ + beta];
            if (!Sab) continue;
            any_nonzero = true;

            // Step 1: allgather full slip vector from PETSc's layout
            {
                const PetscScalar* s_arr;
                CHKERRTHROW(VecGetArrayRead(slip_spatial_[beta], &s_arr));
                CHKERRTHROW(MPI_Allgatherv(s_arr, petsc_counts[my_rank], MPI_DOUBLE,
                                           x_full.data(), petsc_counts.data(),
                                           petsc_displs.data(), MPI_DOUBLE, comm_));
                CHKERRTHROW(VecRestoreArrayRead(slip_spatial_[beta], &s_arr));
            }

            // Step 2: extract STRUMPACK's expected input slice and call mult
            const auto& cdist = Sab->cdist();
            const auto& rdist = Sab->rdist();
            const int s_in_local  = cdist[my_rank+1] - cdist[my_rank];
            const int s_out_local = rdist[my_rank+1] - rdist[my_rank];

            strumpack::DenseMatrix<double> x_dm(s_in_local, 1);
            std::memcpy(x_dm.data(), x_full.data() + cdist[my_rank],
                        s_in_local * sizeof(double));

            strumpack::DenseMatrix<double> y_tmp(s_out_local, 1);
            Sab->mult(strumpack::Trans::N, x_dm, y_tmp);

            // Step 3: allgather output from STRUMPACK's rdist layout
            std::vector<int> rdist_counts(n_ranks), rdist_displs(n_ranks);
            for (int r = 0; r < n_ranks; ++r) rdist_counts[r] = rdist[r+1] - rdist[r];
            rdist_displs[0] = 0;
            for (int r = 1; r < n_ranks; ++r) rdist_displs[r] = rdist[r];
            CHKERRTHROW(MPI_Allgatherv(y_tmp.data(), s_out_local, MPI_DOUBLE,
                                       y_full.data(), rdist_counts.data(),
                                       rdist_displs.data(), MPI_DOUBLE, comm_));

            // Step 4: extract PETSc's local slice and accumulate into traction
            const int petsc_row_start = petsc_dist_[my_rank];
            for (PetscInt i = 0; i < petsc_local; ++i)
                t_arr[i] += static_cast<PetscScalar>(y_full[petsc_row_start + i]);
        }

        CHKERRTHROW(VecRestoreArray(traction_spatial_[alpha], &t_arr));

        if (!any_nonzero) continue;
        CHKERRTHROW(VecScatterBegin(scatter_t_[alpha], traction_spatial_[alpha], t,
                                    ADD_VALUES, SCATTER_FORWARD));
        CHKERRTHROW(VecScatterEnd  (scatter_t_[alpha], traction_spatial_[alpha], t,
                                    ADD_VALUES, SCATTER_FORWARD));
    }
}

// ---------------------------------------------------------------------------
// total_mem_bytes
// ---------------------------------------------------------------------------

double StrumpackGFOperator::total_mem_bytes() const {
    double total = 0.0;
    for (double b : mem_bytes_ab_) total += b;
    return total;
}

} // namespace tndm
