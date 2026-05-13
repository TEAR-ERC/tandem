// Include the header first so PETSc headers are pulled in and
// PETSC_HAVE_HTOOL is defined before we check it below.
#include "HMatrixGreenFunction.h"
#include "common/PetscUtil.h"

#ifdef PETSC_HAVE_HTOOL

#include <petscmathtool.h>

#ifdef HAVE_PETSC_HTOOL_PRIVATE
#include <mat/impls/htool/htool.hpp>
#endif

#include "hilbert.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace tndm {

// ---------------------------------------------------------------------------
// HTool kernel — reads entries from a spatially permuted dense sub-matrix.
// J and K are ACTUAL global row/col indices into G_perm_ab.
// ---------------------------------------------------------------------------
namespace {

struct GFKernelCtxHtool {
    Mat      G;
    PetscInt rstart;
    PetscInt rend;
};

PetscErrorCode GFKernelHtool(PetscInt /*sdim*/, PetscInt M_block, PetscInt N_block,
                              const PetscInt* J, const PetscInt* K,
                              PetscScalar* ptr, void* ctx) {
    auto* kctx = static_cast<GFKernelCtxHtool*>(ctx);
    for (PetscInt j = 0; j < M_block; ++j) {
        for (PetscInt k = 0; k < N_block; ++k) {
            PetscScalar val = 0.0;
            if (J[j] >= kctx->rstart && J[j] < kctx->rend) {
                PetscCall(MatGetValues(kctx->G, 1, &J[j], 1, &K[k], &val));
            }
            ptr[j + M_block * k] = val;
        }
    }
    return PETSC_SUCCESS;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

HMatrixGreenFunction::HMatrixGreenFunction(
    Mat G_dense,
    const std::vector<PetscReal>& local_coords,
    PetscInt nbf, int D,
    Vec s_proto, Vec t_proto,
    MPI_Comm comm,
    HMatrixConfig const& config)
    : D_(D), slip_D_(D - 1), nbf_(nbf), comm_(comm), config_(config)
{
    int n_ranks, my_rank;
    CHKERRTHROW(MPI_Comm_size(comm_, &n_ranks));
    CHKERRTHROW(MPI_Comm_rank(comm_, &my_rank));

    // Derive global element count: M = N_el * D * nbf
    PetscInt M_global, N_global;
    CHKERRTHROW(MatGetSize(G_dense, &M_global, &N_global));
    N_el_ = M_global / (D_ * nbf_);
    assert(N_el_ * D_      * nbf_ == M_global);
    assert(N_el_ * slip_D_ * nbf_ == N_global);

    // ---- Step 1: Allgather local node coords → global (N_el*nbf * D) ----
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

    // ---- Step 2: Spatial permutation (Hilbert curve on scalar N_el*nbf cloud) ----
    build_spatial_permutation(global_coords);

    // ---- Step 3: Build D*(D-1) H-matrices, one (α,β) pair at a time ----
    const int n_ab = D_ * slip_D_;
    H_ab_.resize(n_ab, nullptr);
    mem_bytes_ab_.resize(n_ab, 0.0);

    // When planar_fault=true, α=0 is the normal-traction component. By elastostatics,
    // tangential slip on a planar fault in a homogeneous medium produces zero normal
    // traction. We validate this against the assembled GF and skip the H-matrix build.
    // Collect Frobenius norms first (cheap — just MatNorm, no extra PDE solves).
    std::vector<PetscReal> norms_ab(n_ab, 0.0);
    if (config_.planar_fault) {
        for (int alpha = 0; alpha < D_; ++alpha) {
            for (int beta = 0; beta < slip_D_; ++beta) {
                Mat G_ab = extract_component_submatrix(G_dense, alpha, beta);
                CHKERRTHROW(MatNorm(G_ab, NORM_FROBENIUS, &norms_ab[alpha * slip_D_ + beta]));
                CHKERRTHROW(MatDestroy(&G_ab));
            }
        }
        // Reference norm: max over shear components (α > 0)
        PetscReal ref_norm = 0.0;
        for (int alpha = 1; alpha < D_; ++alpha)
            for (int beta = 0; beta < slip_D_; ++beta)
                ref_norm = std::max(ref_norm, norms_ab[alpha * slip_D_ + beta]);

        if (my_rank == 0 && ref_norm > 0.0) {
            for (int beta = 0; beta < slip_D_; ++beta) {
                double ratio = norms_ab[beta] / static_cast<double>(ref_norm);
                std::cout << "  planar_fault check: ||G_0" << beta << "|| / ||G_shear_max|| = "
                          << ratio << "\n";
            }
        }

        constexpr double PLANAR_FAULT_TOL = 1e-6;
        for (int beta = 0; beta < slip_D_; ++beta) {
            double ratio = (ref_norm > 0.0)
                           ? norms_ab[beta] / static_cast<double>(ref_norm)
                           : 0.0;
            if (ratio > PLANAR_FAULT_TOL) {
                if (my_rank == 0) {
                    std::cerr << "\nERROR: planar_fault=true but normal-traction component G_0"
                              << beta << " is NOT negligible:\n"
                              << "  ||G_0" << beta << "|| / ||G_shear_max|| = " << ratio
                              << "  (tolerance = " << PLANAR_FAULT_TOL << ")\n"
                              << "  This indicates a non-planar fault or inhomogeneous medium.\n"
                              << "  Set planar_fault = false in [hmatrix] to build all components.\n\n";
                }
                throw std::runtime_error(
                    "planar_fault=true validation failed: normal-traction GF is not negligible");
            }
        }
    }

    for (int alpha = 0; alpha < D_; ++alpha) {
        for (int beta = 0; beta < slip_D_; ++beta) {
            const int idx = alpha * slip_D_ + beta;

            // planar_fault=true: skip normal-traction components (α=0), leave H_ab_[idx]=nullptr
            if (config_.planar_fault && alpha == 0) {
                if (my_rank == 0) {
                    std::cout << "  Skipping H_0" << beta
                              << "  (normal traction, zero by planar-fault symmetry)\n";
                }
                continue;
            }

            if (my_rank == 0) {
                std::cout << "  Building H_" << alpha << beta
                          << "  (traction comp " << alpha
                          << " <- slip comp " << beta << ")\n";
            }
            Mat G_ab   = extract_component_submatrix(G_dense, alpha, beta);
            Mat G_perm = build_spatially_permuted_submatrix(G_ab);
            CHKERRTHROW(MatDestroy(&G_ab));
            build_one_h_matrix(alpha, beta, G_perm, global_coords);
            CHKERRTHROW(MatDestroy(&G_perm));
        }
    }

    // ---- Step 4: Per-component scatter objects ----
    build_scatters(s_proto, t_proto);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

HMatrixGreenFunction::~HMatrixGreenFunction() {
    for (auto& H : H_ab_)             MatDestroy(&H);
    for (auto& v : slip_spatial_)     VecDestroy(&v);
    for (auto& v : traction_spatial_) VecDestroy(&v);
    for (auto& s : scatter_s_)        VecScatterDestroy(&s);
    for (auto& s : scatter_t_)        VecScatterDestroy(&s);
}

// ---------------------------------------------------------------------------
// PCA eigensolvers — used by build_spatial_permutation to detect the fault's
// intrinsic dimension and rotate coordinates to the fault's principal axes.
//
// Convention: eval[0] >= eval[1] >= ..., evec[k*D+d] = component d of the
// k-th principal direction (row-major, one direction per row).
// ---------------------------------------------------------------------------

static void sym2_eigen(const double* A, double* eval, double* evec) {
    // Closed-form solution for 2×2 symmetric [[a,b],[b,d]]
    double a = A[0], b = A[1], d = A[3];
    double tr   = a + d;
    double disc = std::sqrt(std::max(0.0, 0.25*(a-d)*(a-d) + b*b));
    eval[0] = 0.5*tr + disc;
    eval[1] = 0.5*tr - disc;

    if (std::abs(b) > 1e-14) {
        double v0x = b, v0y = eval[0] - a;
        double len = std::sqrt(v0x*v0x + v0y*v0y);
        evec[0] =  v0x/len;  evec[1] = v0y/len;   // eigenvec 0 (row 0)
        evec[2] = -v0y/len;  evec[3] = v0x/len;   // eigenvec 1 (row 1, perpendicular)
    } else if (a >= d) {
        evec[0]=1; evec[1]=0; evec[2]=0; evec[3]=1;
    } else {
        evec[0]=0; evec[1]=1; evec[2]=1; evec[3]=0;
    }
}

static void sym3_eigen(const double* A, double* eval, double* evec) {
    // Jacobi iteration for 3×3 symmetric matrix.
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

    // Sort indices by descending diagonal (eigenvalue)
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
// build_spatial_permutation — PCA-based Hilbert ordering
//
// Algorithm:
//  1. PCA of the node coordinate cloud → eigenvalues λ₀ ≥ λ₁ ≥ λ₂ and
//     the corresponding principal directions (eigenvectors).
//  2. Effective dimension: keep directions where λₖ/λ₀ > leaf_size/Np.
//     Single planar fault  → λ_normal≈0   → eff_dim=2 (2-D Hilbert)
//     Single straight fault → λ_trans≈0   → eff_dim=1 (1-D sort)
//     Multiple/curved faults → all λ large → eff_dim=3 (3-D Hilbert)
//  3. Project coordinates onto the eff_dim principal directions.
//  4. Isotropic (cubic) normalisation: scale all axes by the same factor
//     (max extent across directions) → correct physical distance weighting.
//  5. Apply Hilbert curve of dimension eff_dim from hilbert::v2.
//
// This avoids the start/end adjacency problem that arises when a 3-D Hilbert
// curve is restricted to a flat 2-D surface (the start and end happen to be
// physically adjacent on the flat surface), and naturally handles complex
// fault geometries — multiple faults, curved surfaces, etc. — without any
// hard-coded dimension assumption.
// ---------------------------------------------------------------------------

void HMatrixGreenFunction::build_spatial_permutation(
    const std::vector<PetscReal>& global_coords)
{
    const PetscInt Np = N_el_ * nbf_;
    perm_.resize(Np);
    std::iota(perm_.begin(), perm_.end(), 0);

    // ---- 1. Centroid ----
    std::vector<double> mean(D_, 0.0);
    for (PetscInt i = 0; i < Np; ++i)
        for (int d = 0; d < D_; ++d)
            mean[d] += static_cast<double>(global_coords[i*D_+d]);
    for (int d = 0; d < D_; ++d) mean[d] /= static_cast<double>(Np);

    // ---- 2. Covariance matrix (D×D, symmetric) ----
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

    // ---- 3. Eigendecomposition ----
    std::vector<double> eval(D_), evec(D_*D_);
    if (D_ == 2) sym2_eigen(cov.data(), eval.data(), evec.data());
    else         sym3_eigen(cov.data(), eval.data(), evec.data());
    // eval[0] >= eval[1] >= eval[2]; evec[k*D_+d] = d-th component of k-th direction

    // ---- 4. Effective dimension ----
    // A direction is degenerate when its variance is < (leaf_size/Np) × max_variance.
    // leaf_size/Np ≈ (cluster_DOF_count / total_DOF_count) — the fraction of the
    // total point spread occupied by one cluster.
    const double thr = static_cast<double>(config_.leaf_size) / static_cast<double>(Np);
    int eff_dim = 0;
    for (int d = 0; d < D_; ++d)
        if (eval[0] > 0.0 && eval[d] / eval[0] > thr) ++eff_dim;
    if (eff_dim < 1) eff_dim = 1;  // safety fallback

    // Print on rank 0 so the user can see which dimension was chosen
    {
        int rank; MPI_Comm_rank(comm_, &rank);
        if (rank == 0) {
            std::cout << "  Hilbert ordering: eff_dim=" << eff_dim
                      << "  eigenvalue ratios [";
            for (int d = 0; d < D_; ++d) {
                std::cout << (eval[0]>0.0 ? eval[d]/eval[0] : 0.0);
                if (d < D_-1) std::cout << ", ";
            }
            std::cout << "]  threshold=" << thr << "\n";
        }
    }

    // ---- 5. Project to eff_dim principal directions ----
    std::vector<double> proj(Np * eff_dim);
    for (PetscInt i = 0; i < Np; ++i)
        for (int k = 0; k < eff_dim; ++k) {
            double p = 0.0;
            for (int d = 0; d < D_; ++d)
                p += (static_cast<double>(global_coords[i*D_+d]) - mean[d]) * evec[k*D_+d];
            proj[i*eff_dim+k] = p;
        }

    // ---- 6. Isotropic (cubic) normalisation to [0, UINT16_MAX] ----
    // Use the same scale L for all dimensions so that physical distances are
    // preserved — a cluster diameter of d physical units maps to the same
    // number of grid cells regardless of which axis it is on.
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

    // ---- 7. Sort perm_ by Hilbert index of dimension eff_dim ----
    if (eff_dim == 1) {
        // 1-D: plain ascending sort by the projected coordinate
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

    // Diagnostic: print physical coordinates of the first and last nodes so we can
    // verify that the Hilbert ordering places maximally-separated nodes at the endpoints.
    {
        int rank; MPI_Comm_rank(comm_, &rank);
        if (rank == 0) {
            auto print_node = [&](const char* label, PetscInt perm_pos) {
                PetscInt node = perm_[perm_pos];
                std::cout << "  Hilbert " << label << " (perm[" << perm_pos << "]=node "
                          << node << "): (";
                for (int d = 0; d < D_; ++d) {
                    std::cout << global_coords[node * D_ + d];
                    if (d < D_-1) std::cout << ", ";
                }
                // also print projected coords
                std::cout << ")  proj=(";
                for (int k = 0; k < eff_dim; ++k) {
                    std::cout << proj[node * eff_dim + k];
                    if (k < eff_dim-1) std::cout << ", ";
                }
                std::cout << ")\n";
            };
            print_node("first  ", 0);
            print_node("last   ", Np - 1);
            // Euclidean distance between first and last in 3D physical coords
            double dist = 0.0;
            for (int d = 0; d < D_; ++d) {
                double dv = static_cast<double>(global_coords[perm_[0]*D_+d])
                          - static_cast<double>(global_coords[perm_[Np-1]*D_+d]);
                dist += dv*dv;
            }
            std::cout << "  Hilbert start→end physical dist = " << std::sqrt(dist)
                      << "  (fault extent L=" << L << ")\n";
        }
    }
}

// ---------------------------------------------------------------------------
// extract_component_submatrix — build G_αβ : (N_el*nbf) × (N_el*nbf)
// Reads rows for traction component α and columns for slip component β from G_dense.
// ---------------------------------------------------------------------------

Mat HMatrixGreenFunction::extract_component_submatrix(Mat G_dense, int alpha, int beta) const {
    PetscInt Np        = N_el_ * nbf_;
    PetscInt row_block = D_     * nbf_;   // rows per element in G_dense
    PetscInt col_block = slip_D_ * nbf_;  // cols per element in G_dense

    // Column index mapping: scalar col j → G_dense col
    std::vector<PetscInt> src_cols(Np), dst_cols(Np);
    for (PetscInt e = 0; e < N_el_; ++e) {
        for (PetscInt n = 0; n < nbf_; ++n) {
            PetscInt j      = e * nbf_ + n;
            src_cols[j]     = e * col_block + beta * nbf_ + n;
            dst_cols[j]     = j;
        }
    }

    // G_dense row ownership (elements are blocked by row_block rows)
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
        CHKERRTHROW(MatGetValues(G_dense, nbf_, src_rows.data(),
                                 Np, src_cols.data(), block.data()));
        CHKERRTHROW(MatSetValues(G_ab, nbf_, dst_rows.data(),
                                 Np, dst_cols.data(), block.data(), INSERT_VALUES));
    }

    CHKERRTHROW(MatAssemblyBegin(G_ab, MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd  (G_ab, MAT_FINAL_ASSEMBLY));
    return G_ab;
}

// ---------------------------------------------------------------------------
// build_spatially_permuted_submatrix — reorder rows and cols of G_ab with perm_
// G_perm[inv(perm_[i]), inv(perm_[j])] = G_ab[i, j]
// ---------------------------------------------------------------------------

Mat HMatrixGreenFunction::build_spatially_permuted_submatrix(Mat G_ab) const {
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
        CHKERRTHROW(MatSetValues(G_perm, 1, &new_i, Np,
                                 all_new_cols.data(), row_new.data(), INSERT_VALUES));
    }
    CHKERRTHROW(MatAssemblyBegin(G_perm, MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd  (G_perm, MAT_FINAL_ASSEMBLY));
    return G_perm;
}

// ---------------------------------------------------------------------------
// build_one_h_matrix — assemble a single HTool H-matrix from G_perm_ab
// ---------------------------------------------------------------------------

void HMatrixGreenFunction::build_one_h_matrix(int alpha, int beta, Mat G_perm_ab,
                                               const std::vector<PetscReal>& global_coords)
{
    int n_ranks, my_rank;
    CHKERRTHROW(MPI_Comm_size(comm_, &n_ranks));
    CHKERRTHROW(MPI_Comm_rank(comm_, &my_rank));

    PetscInt Np = N_el_ * nbf_;

    {
        long long required = static_cast<long long>(n_ranks) * config_.leaf_size;
        if (required > static_cast<long long>(Np)) {
            CHKERRTHROW(PetscPrintf(comm_,
                "\nERROR: H-matrix G_%d%d: nranks=%d, leaf_size=%d"
                " => nranks*leaf_size=%lld > Np=%lld\n\n",
                alpha, beta, n_ranks, config_.leaf_size, required, (long long)Np));
            throw std::runtime_error("H-matrix: nranks * leaf_size > Np");
        }
    }

    // Balanced row distribution (from PETSC_DECIDE on G_perm_ab)
    PetscInt perm_rstart, perm_rend;
    CHKERRTHROW(MatGetOwnershipRange(G_perm_ab, &perm_rstart, &perm_rend));
    const PetscInt local_m = perm_rend - perm_rstart;

    // Balanced column distribution (mirrors PETSc PETSC_DECIDE formula)
    const PetscInt col_base  = Np / n_ranks;
    const PetscInt col_extra = Np % n_ranks;
    const PetscInt perm_cstart = my_rank * col_base + std::min(my_rank, (int)col_extra);
    const PetscInt perm_cend   = perm_cstart + col_base + (my_rank < (int)col_extra ? 1 : 0);
    const PetscInt local_n     = perm_cend - perm_cstart;

    // Local target coords: perm_-ordered coords for rows [perm_rstart, perm_rend)
    std::vector<PetscReal> local_target(local_m * D_);
    for (PetscInt ni = perm_rstart; ni < perm_rend; ++ni) {
        PetscInt old_i = perm_[ni];
        for (int d = 0; d < D_; ++d)
            local_target[(ni - perm_rstart) * D_ + d] = global_coords[old_i * D_ + d];
    }

    // Local source coords: perm_-ordered coords for cols [perm_cstart, perm_cend)
    std::vector<PetscReal> local_source(local_n * D_);
    for (PetscInt nj = perm_cstart; nj < perm_cend; ++nj) {
        PetscInt old_j = perm_[nj];
        for (int d = 0; d < D_; ++d)
            local_source[(nj - perm_cstart) * D_ + d] = global_coords[old_j * D_ + d];
    }

    GFKernelCtxHtool kctx{G_perm_ab, perm_rstart, perm_rend};

    CHKERRTHROW(PetscOptionsSetValue(nullptr, "-mat_htool_epsilon",
                                     std::to_string(config_.rtol).c_str()));
    CHKERRTHROW(PetscOptionsSetValue(nullptr, "-mat_htool_eta",
                                     std::to_string(config_.eta).c_str()));
    CHKERRTHROW(PetscOptionsSetValue(nullptr, "-mat_htool_min_cluster_size",
                                     std::to_string(config_.leaf_size).c_str()));

    Mat& H = H_ab_[alpha * slip_D_ + beta];
    CHKERRTHROW(MatCreateHtoolFromKernel(
        comm_,
        local_m, local_n,
        Np, Np,
        static_cast<PetscInt>(D_),
        local_target.data(),
        local_source.data(),
        GFKernelHtool, &kctx,
        &H));
    CHKERRTHROW(MatSetOption(H, MAT_SYMMETRIC, PETSC_FALSE));
    CHKERRTHROW(MatSetFromOptions(H));
    CHKERRTHROW(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd  (H, MAT_FINAL_ASSEMBLY));

#ifdef HAVE_PETSC_HTOOL_PRIVATE
    {
        Mat_Htool* impl = nullptr;
        CHKERRTHROW(MatShellGetContext(H, &impl));
        auto info = htool::get_distributed_hmatrix_information(
            impl->distributed_operator_holder->hmatrix, comm_);
        auto it = info.find("Compression_ratio");
        if (it != info.end()) {
            double cr = std::stod(it->second);
            if (cr > 0.0) {
                mem_bytes_ab_[alpha * slip_D_ + beta] =
                    static_cast<double>(Np) * static_cast<double>(Np)
                    * sizeof(PetscScalar) / cr;
            }
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// build_scatters — create per-component VecScatter objects
// ---------------------------------------------------------------------------

void HMatrixGreenFunction::build_scatters(Vec s_proto, Vec t_proto) {
    PetscInt Np = N_el_ * nbf_;

    // Get balanced distribution from the first non-null H_ab_.
    // (planar_fault=true leaves some entries nullptr, so we can't always use index 0.)
    Mat H_ref = nullptr;
    for (Mat H : H_ab_) { if (H != nullptr) { H_ref = H; break; } }
    if (H_ref == nullptr) {
        throw std::runtime_error("HMatrixGreenFunction: all H_ab_ entries are null — "
                                 "nothing to build scatters for");
    }
    Vec tmp_col, tmp_row;
    CHKERRTHROW(MatCreateVecs(H_ref, &tmp_col, &tmp_row));
    PetscInt perm_cstart, perm_cend, perm_rstart, perm_rend;
    CHKERRTHROW(VecGetOwnershipRange(tmp_col, &perm_cstart, &perm_cend));
    CHKERRTHROW(VecGetOwnershipRange(tmp_row, &perm_rstart, &perm_rend));
    const PetscInt local_n = perm_cend - perm_cstart;
    const PetscInt local_m = perm_rend - perm_rstart;
    CHKERRTHROW(VecDestroy(&tmp_col));
    CHKERRTHROW(VecDestroy(&tmp_row));

    // Allocate per-component work vectors
    slip_spatial_.resize(slip_D_, nullptr);
    traction_spatial_.resize(D_, nullptr);
    for (int beta = 0; beta < slip_D_; ++beta)
        CHKERRTHROW(VecCreateMPI(comm_, local_n, Np, &slip_spatial_[beta]));
    for (int alpha = 0; alpha < D_; ++alpha)
        CHKERRTHROW(VecCreateMPI(comm_, local_m, Np, &traction_spatial_[alpha]));

    // Slip scatters: s β-component slice (original) → slip_spatial_[β]
    // from_idx[nj - cstart] = index in s_proto for spatial column nj, component β
    scatter_s_.resize(slip_D_, nullptr);
    {
        std::vector<PetscInt> from_idx(local_n);
        for (int beta = 0; beta < slip_D_; ++beta) {
            for (PetscInt nj = perm_cstart; nj < perm_cend; ++nj) {
                PetscInt scalar = perm_[nj];
                PetscInt e = scalar / nbf_;
                PetscInt n = scalar % nbf_;
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
    // to_idx[ni - rstart] = index in t_proto for spatial row ni, component α
    scatter_t_.resize(D_, nullptr);
    {
        std::vector<PetscInt> to_idx(local_m);
        for (int alpha = 0; alpha < D_; ++alpha) {
            for (PetscInt ni = perm_rstart; ni < perm_rend; ++ni) {
                PetscInt scalar = perm_[ni];
                PetscInt e = scalar / nbf_;
                PetscInt n = scalar % nbf_;
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
// apply — t += Σ_{α,β} H_αβ * s_β   (both in original Tandem DOF order)
// ---------------------------------------------------------------------------

void HMatrixGreenFunction::apply(Vec s, Vec t) const {
    // Scatter all slip components to spatial order (once per β, reused for all α)
    for (int beta = 0; beta < slip_D_; ++beta) {
        CHKERRTHROW(VecScatterBegin(scatter_s_[beta], s, slip_spatial_[beta],
                                    INSERT_VALUES, SCATTER_FORWARD));
        CHKERRTHROW(VecScatterEnd  (scatter_s_[beta], s, slip_spatial_[beta],
                                    INSERT_VALUES, SCATTER_FORWARD));
    }

    for (int alpha = 0; alpha < D_; ++alpha) {
        CHKERRTHROW(VecZeroEntries(traction_spatial_[alpha]));
        // Accumulate: t_α_spatial += H_αβ * s_β_spatial  for each β
        // H_ab_[idx] may be nullptr when planar_fault=true skipped that component.
        bool any_nonzero = false;
        for (int beta = 0; beta < slip_D_; ++beta) {
            Mat H = H_ab_[alpha * slip_D_ + beta];
            if (H == nullptr) continue;
            any_nonzero = true;
            CHKERRTHROW(MatMultAdd(H, slip_spatial_[beta],
                                   traction_spatial_[alpha],
                                   traction_spatial_[alpha]));
        }
        // Only scatter back if there is a non-zero contribution (avoids spurious ADD_VALUES)
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

double HMatrixGreenFunction::total_mem_bytes() const {
    double total = 0.0;
    for (double b : mem_bytes_ab_) total += b;
    return total;
}

// ---------------------------------------------------------------------------
// export_structure
// ---------------------------------------------------------------------------

void HMatrixGreenFunction::export_structure(const std::string& prefix) const {
#ifdef HAVE_PETSC_HTOOL_PRIVATE
    for (int alpha = 0; alpha < D_; ++alpha) {
        for (int beta = 0; beta < slip_D_; ++beta) {
            Mat H = H_ab_[alpha * slip_D_ + beta];
            if (H == nullptr) {
                int rank;
                MPI_Comm_rank(comm_, &rank);
                if (rank == 0) {
                    std::cout << "  Skipping export for H_" << alpha << beta
                              << " (not built — planar_fault=true)\n";
                }
                continue;
            }
            std::string sub = prefix + "_ab"
                              + std::to_string(alpha) + std::to_string(beta);
            export_one_h_structure(H, sub);
        }
    }
#else
    int rank;
    MPI_Comm_rank(comm_, &rank);
    if (rank == 0) {
        std::cout << "export_structure: PETSC_SRC_DIR not set at build time; "
                     "cannot access internal HTool HMatrix. Skipping.\n";
    }
    (void)prefix;
#endif
}

#ifdef HAVE_PETSC_HTOOL_PRIVATE
void HMatrixGreenFunction::export_one_h_structure(Mat H, const std::string& sub) const {
    int rank, nranks;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &nranks);

    Mat_Htool* impl = nullptr;
    CHKERRTHROW(MatShellGetContext(H, &impl));
    const auto& hmat = impl->distributed_operator_holder->hmatrix;

    if (rank == 0) {
        std::cout << "\n=== H-matrix structure: " << sub << " ===\n";
        htool::print_tree_parameters(hmat, std::cout);
    }
    htool::print_distributed_hmatrix_information(hmat, std::cout, comm_);

    // Per-rank CSV with local offsets (native HTool format)
    htool::save_leaves_with_rank(hmat, sub + "_rank" + std::to_string(rank));

    // Gather global-offset leaf data to rank 0
    using HM = htool::HMatrix<PetscScalar, PetscReal>;
    struct Leaf { int row0, nrows, col0, ncols, crank; };
    std::vector<Leaf> local_leaves;

    htool::preorder_tree_traversal(hmat, [&local_leaves](const HM& node) {
        if (node.is_leaf()) {
            local_leaves.push_back({
                static_cast<int>(node.get_target_cluster().get_offset()),
                static_cast<int>(node.get_target_cluster().get_size()),
                static_cast<int>(node.get_source_cluster().get_offset()),
                static_cast<int>(node.get_source_cluster().get_size()),
                node.get_rank()
            });
        }
    });

    int local_count = static_cast<int>(local_leaves.size());
    std::vector<int> counts(nranks);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm_);

    constexpr int FIELDS = 5;
    std::vector<int> flat(local_count * FIELDS);
    for (int i = 0; i < local_count; ++i) {
        flat[i*FIELDS+0] = local_leaves[i].row0;
        flat[i*FIELDS+1] = local_leaves[i].nrows;
        flat[i*FIELDS+2] = local_leaves[i].col0;
        flat[i*FIELDS+3] = local_leaves[i].ncols;
        flat[i*FIELDS+4] = local_leaves[i].crank;
    }

    std::vector<int> recv_counts(nranks), rdispls(nranks, 0);
    for (int r = 0; r < nranks; ++r) recv_counts[r] = counts[r] * FIELDS;
    for (int r = 1; r < nranks; ++r) rdispls[r] = rdispls[r-1] + recv_counts[r-1];
    int total = rdispls[nranks-1] + recv_counts[nranks-1];

    std::vector<int> all_leaves;
    if (rank == 0) all_leaves.resize(total);
    MPI_Gatherv(flat.data(), local_count * FIELDS, MPI_INT,
                all_leaves.data(), recv_counts.data(), rdispls.data(), MPI_INT, 0, comm_);

    if (rank == 0) {
        PetscInt M, N;
        MatGetSize(H, &M, &N);
        std::ofstream out(sub + ".csv");
        out << M << "," << N << "\n";
        int n_leaves = total / FIELDS;
        for (int i = 0; i < n_leaves; ++i) {
            out << all_leaves[i*FIELDS+0] << ","
                << all_leaves[i*FIELDS+1] << ","
                << all_leaves[i*FIELDS+2] << ","
                << all_leaves[i*FIELDS+3] << ","
                << all_leaves[i*FIELDS+4] << "\n";
        }
        std::cout << "Wrote " << n_leaves << " leaves to " << sub << ".csv\n";
    }
}
#endif // HAVE_PETSC_HTOOL_PRIVATE

} // namespace tndm

#endif // PETSC_HAVE_HTOOL
