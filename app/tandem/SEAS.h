#ifndef SEAS_20200825_H
#define SEAS_20200825_H

#include "config.h"
#include "tandem/SeasConfig.h"

#include "mesh/LocalSimplexMesh.h"

#include <string>

namespace tndm {
void solveSEASProblem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg);
// Builds G_ and H_, applies both to a random vector, prints and returns relative error.
// Forces use_hmatrix = true. Returns -1.0 if HTool is unavailable.
double validateGFHMatrix(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg);
// Builds H_ and exports its leaf structure to <output_prefix>.csv (merged, global offsets)
// and <output_prefix>_rank<r>.csv (per-rank, local offsets). Forces use_hmatrix = true.
void dumpGFHMatrixStructure(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg,
                            const std::string& output_prefix);

struct SolveBenchResult {
    int    n_ranks    = 0;
    int    n_reps     = 0;
    double time_avg_s = -1.0;
    double time_min_s = -1.0;
    double time_max_s = -1.0;
};
// Sets up the KSP solver (mesh + DG operator only, no GF construction), then runs
// nreps timed calls to solve(v)+update_traction with a fixed random rhs.
// One warmup rep is performed before the timed loop.
SolveBenchResult benchmarkSolve(LocalSimplexMesh<DomainDimension> const& mesh,
                                Config const& cfg, int nreps);
}

#endif // SEAS_20200825_H
