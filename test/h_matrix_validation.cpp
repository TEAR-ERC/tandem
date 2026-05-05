/**
 * H-matrix validation test.
 *
 * Validates that the H2Opus H-matrix constructed via MatCreateH2OpusFromKernel
 * produces matrix-vector products consistent with a reference dense matrix.
 *
 * This mirrors the API path used by the actual implementation in
 * SeasQDDiscreteGreenOperator::build_h_matrix().
 *
 * Strategy:
 *   1. Define a Gaussian kernel K(x,y) = exp(-||x-y||^2 / sigma^2) whose
 *      smooth, rapidly-decaying character gives low off-diagonal rank —
 *      ideal for H-matrix approximation.
 *   2. Build a reference MATSEQDENSE matrix by evaluating the kernel for
 *      every (i,j) pair directly.
 *   3. Call MatCreateH2OpusFromKernel with the same kernel and point coordinates.
 *      Disable the H2Opus symmetry check via PetscOptionsSetValue before assembly.
 *   4. Generate a reproducible random vector x (VecSetRandom, seed 12345).
 *   5. Compute y_dense = A * x and y_H = H * x.
 *   6. Check relative error ||y_dense - y_H|| / ||y_dense|| < 1e-3.
 *
 * PETSc is initialised once in main() so that PetscInitialize/PetscFinalize
 * are not called inside individual TEST_CASEs.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <mpi.h>
#include <petscsys.h>
#include <petscmat.h>
#include <petscvec.h>

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Custom main: initialise MPI + PETSc, run doctest, finalise.
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    int res = doctest::Context(argc, argv).run();

    PetscFinalize();
    MPI_Finalize();
    return res;
}

// ============================================================
// Implementation - only compiled when H2Opus is available
// ============================================================
#ifdef PETSC_HAVE_H2OPUS

namespace {

// ---------------------------------------------------------------------------
// Context passed to the kernel function.
// ---------------------------------------------------------------------------
struct GaussianKernelCtx {
    PetscReal sigma2; // sigma^2 — width parameter
};

// ---------------------------------------------------------------------------
// Gaussian kernel matching the MatH2OpusKernelFn typedef:
//   PetscScalar fn(PetscInt sdim, PetscReal x[], PetscReal y[], void *ctx)
// Computes exp(-||x-y||^2 / sigma^2).
// ---------------------------------------------------------------------------
PetscScalar GaussianKernel(PetscInt sdim, PetscReal x[], PetscReal y[], void* ctx) {
    const auto* kctx = static_cast<const GaussianKernelCtx*>(ctx);
    PetscReal dist2 = 0.0;
    for (PetscInt d = 0; d < sdim; ++d) {
        const PetscReal diff = x[d] - y[d];
        dist2 += diff * diff;
    }
    return std::exp(-dist2 / kctx->sigma2);
}

// ---------------------------------------------------------------------------
// Build a square MATSEQDENSE reference matrix from the Gaussian kernel.
// coords layout: [point_index * sdim + dim_index].
// ---------------------------------------------------------------------------
Mat build_dense_matrix(PetscInt N, PetscInt sdim, const std::vector<PetscReal>& coords,
                        GaussianKernelCtx& kctx) {
    Mat A;
    MatCreateSeqDense(PETSC_COMM_SELF, N, N, nullptr, &A);
    for (PetscInt i = 0; i < N; ++i) {
        for (PetscInt j = 0; j < N; ++j) {
            // const_cast is safe: kernel does not modify the coordinate arrays
            PetscReal* xi = const_cast<PetscReal*>(&coords[i * sdim]);
            PetscReal* xj = const_cast<PetscReal*>(&coords[j * sdim]);
            PetscScalar val = GaussianKernel(sdim, xi, xj, &kctx);
            MatSetValue(A, i, j, val, INSERT_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    return A;
}

// ---------------------------------------------------------------------------
// Uniform 1-D coordinates in [0, L].
// ---------------------------------------------------------------------------
std::vector<PetscReal> uniform_coords_1d(PetscInt N, PetscReal L = 10.0) {
    std::vector<PetscReal> coords(N);
    for (PetscInt i = 0; i < N; ++i) {
        coords[i] =
            (N > 1) ? L * static_cast<PetscReal>(i) / static_cast<PetscReal>(N - 1) : 0.0;
    }
    return coords;
}

// ---------------------------------------------------------------------------
// Uniform 2-D grid coordinates in [0,L] x [0,L].  Returns nx*nx points.
// Layout: [point_index * 2 + dim_index].
// ---------------------------------------------------------------------------
std::vector<PetscReal> uniform_coords_2d(PetscInt nx, PetscReal L = 5.0) {
    const PetscInt N = nx * nx;
    std::vector<PetscReal> coords(N * 2);
    PetscInt idx = 0;
    for (PetscInt i = 0; i < nx; ++i) {
        const PetscReal xi =
            (nx > 1) ? L * static_cast<PetscReal>(i) / static_cast<PetscReal>(nx - 1) : 0.0;
        for (PetscInt j = 0; j < nx; ++j) {
            const PetscReal xj =
                (nx > 1) ? L * static_cast<PetscReal>(j) / static_cast<PetscReal>(nx - 1) : 0.0;
            coords[idx * 2 + 0] = xi;
            coords[idx * 2 + 1] = xj;
            ++idx;
        }
    }
    return coords;
}

// ---------------------------------------------------------------------------
// Core validation:
//   - Builds an H-matrix via MatCreateH2OpusFromKernel (same API as the real
//     implementation in SeasQDDiscreteGreenOperator::build_h_matrix()).
//   - Multiplies both the dense reference and the H-matrix by the same random
//     vector and returns the relative error.
// ---------------------------------------------------------------------------
double run_validation(Mat A_dense, PetscInt N, PetscInt sdim,
                       const std::vector<PetscReal>& coords, GaussianKernelCtx& kctx) {
    PetscErrorCode ierr;

    // Disable the H2Opus internal symmetry check.  Must happen BEFORE
    // MatAssemblyEnd so that the option is active during construction.
    // (This mirrors the call in SeasQDDiscreteGreenOperator::build_h_matrix().)
    ierr = PetscOptionsSetValue(NULL, "-mat_h2opus_check", "false");
    REQUIRE(ierr == 0);

    // Build H-matrix via kernel — this is the production code path.
    Mat H;
    ierr = MatCreateH2OpusFromKernel(
        PETSC_COMM_SELF,
        N, N,         // local rows, local cols
        N, N,         // global rows, global cols
        sdim,         // spatial dimension
        coords.data(), // point coordinates [N * sdim]
        PETSC_FALSE,  // cdist: coordinates are NOT distributed
        GaussianKernel, &kctx,
        0.9,          // eta: admissibility parameter
        32,           // leafsize: cluster leaf size
        8,            // basisord: basis polynomial order
        &H);
    REQUIRE(ierr == 0);

    ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
    REQUIRE(ierr == 0);
    ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);
    REQUIRE(ierr == 0);

    // Create vectors.
    Vec x, y_dense, y_H;
    VecCreateSeq(PETSC_COMM_SELF, N, &x);
    VecDuplicate(x, &y_dense);
    VecDuplicate(x, &y_H);

    // Fill x with reproducible pseudo-random values.
    PetscRandom rng;
    PetscRandomCreate(PETSC_COMM_SELF, &rng);
    PetscRandomSetType(rng, PETSCRAND48);
    PetscRandomSetSeed(rng, 12345UL);
    PetscRandomSeed(rng);
    VecSetRandom(x, rng);
    PetscRandomDestroy(&rng);

    // Dense reference multiply.
    ierr = MatMult(A_dense, x, y_dense);
    REQUIRE(ierr == 0);

    // H-matrix multiply.
    ierr = MatMult(H, x, y_H);
    REQUIRE(ierr == 0);

    // Relative error: ||y_H - y_dense|| / ||y_dense||.
    VecAXPY(y_H, -1.0, y_dense); // y_H <- y_H - y_dense  (error vector)
    PetscReal err_norm = 0.0, ref_norm = 1.0;
    VecNorm(y_H,     NORM_2, &err_norm);
    VecNorm(y_dense, NORM_2, &ref_norm);

    double rel_err =
        (static_cast<double>(ref_norm) > 0.0)
            ? static_cast<double>(err_norm) / static_cast<double>(ref_norm)
            : 0.0;

    VecDestroy(&x);
    VecDestroy(&y_dense);
    VecDestroy(&y_H);
    MatDestroy(&H);

    return rel_err;
}

} // anonymous namespace

// ============================================================
// Test cases
// ============================================================

TEST_CASE("H-matrix 1D Gaussian kernel N=64") {
    const PetscInt  N      = 64;
    const PetscInt  sdim   = 1;
    GaussianKernelCtx kctx{4.0}; // sigma = 2.0  =>  sigma^2 = 4.0
    const double    tol    = 1e-3;

    auto coords = uniform_coords_1d(N, 10.0);
    Mat  A      = build_dense_matrix(N, sdim, coords, kctx);

    double rel_err = run_validation(A, N, sdim, coords, kctx);
    MatDestroy(&A);

    INFO("1D N=" << N << " relative error=" << rel_err << " tol=" << tol);
    CHECK(rel_err < tol);
}

TEST_CASE("H-matrix 2D Gaussian kernel N=100 (10x10 grid)") {
    const PetscInt  nx   = 10;
    const PetscInt  N    = nx * nx;
    const PetscInt  sdim = 2;
    GaussianKernelCtx kctx{1.0}; // sigma = 1.0  =>  sigma^2 = 1.0
    const double    tol  = 1e-3;

    auto coords = uniform_coords_2d(nx, 5.0);
    Mat  A      = build_dense_matrix(N, sdim, coords, kctx);

    double rel_err = run_validation(A, N, sdim, coords, kctx);
    MatDestroy(&A);

    INFO("2D N=" << N << " relative error=" << rel_err << " tol=" << tol);
    CHECK(rel_err < tol);
}

#else  // !PETSC_HAVE_H2OPUS

TEST_CASE("H-matrix validation (H2Opus not available - skipped)") {
    // H2Opus support was not compiled into PETSc; nothing to validate.
    // Mark as passing so the overall test suite is not broken.
    MESSAGE("Skipping: PETSc was not built with H2Opus (PETSC_HAVE_H2OPUS undefined).");
    CHECK(true);
}

#endif // PETSC_HAVE_H2OPUS
