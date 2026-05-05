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
}

#endif // SEAS_20200825_H
