#ifndef HMATRIXCONFIG_20260501_H
#define HMATRIXCONFIG_20260501_H

#include <string>

namespace tndm {

struct HMatrixConfig {
    bool use_hmatrix = false;
    double eta = 0.9;
    int leaf_size = 32;
    int basis_order = 8;
    // max_rank and batch_size are reserved for future MatCreateH2OpusFromMat path.
    // NOTE: max_rank has NO effect on the STRUMPACK path. It is forwarded as far as
    // HODLROptions, but HODLRMatrix never reads opts.max_rank() and ButterflyPACK is given
    // no rank cap — only "rank0" (an initial guess) from rank_guess below. Observed block
    // ranks routinely exceed this value; do not use it expecting a cap.
    int max_rank = 64;
    int batch_size = 32;
    double rtol = 1e-4;
    // --- ButterflyPACK knobs, reachable only via HODLR::HODLROptions ---
    // These are dropped by structured::construct_matrix_free (its StructuredOptions ->
    // HODLROptions conversion keeps only the base fields), so StrumpackGFFullOperator
    // constructs HODLR::HODLRMatrix directly in order to pass them.
    //
    // less_adapt = true (ButterflyPACK default) adapts the block rank only at the coarsest
    // levels and extrapolates for deeper ones. That is visible in the build log as Ntrial>1
    // at level ~4 and Ntrial==1 at every deeper level, with deep-level errors settling well
    // above rtol. Set false to make every level adapt to rtol, at the cost of more matvecs.
    bool less_adapt = true;
    int rank_guess = 128;       // ButterflyPACK "rank0" — initial rank guess
    double rank_rate = 2.0;     // ButterflyPACK "rankrate" — rank growth factor per trial
    double bf_sampling = 1.2;   // ButterflyPACK "sample_para" / "sample_para_outer"
    // Compressed format: "hodlr" (low-rank off-diagonal blocks) or "hodbf" (butterfly
    // off-diagonal blocks; better for the O(sqrt(N)) ranks a 2D fault surface produces).
    std::string format = "hodlr";
    // Cluster tree used by the STRUMPACK matrix-free operator:
    //   "petsc1d" — binary bisection matching PETSc's PETSC_DECIDE distribution (default;
    //               keeps the proven bp3 behaviour).
    //   "kdtree"  — 2D/3D k-d (median-split) spatial tree that induces perm_ and recurses to
    //               leaf_size, so off-diagonal HODLR blocks stay spatially separated at every
    //               level. Needed for 2D-fault (3D) problems such as bp7.
    std::string cluster_tree = "petsc1d";
    // For planar faults in a homogeneous elastic medium, tangential slip produces
    // zero normal traction on the fault plane (exact result from elastostatics).
    // Set planar_fault = true to skip building H-matrices for those zero components,
    // with validation that they are indeed negligible in the assembled dense GF.
    bool planar_fault = false;
};

} // namespace tndm

#endif // HMATRIXCONFIG_20260501_H
