#ifndef SEAS_20200825_H
#define SEAS_20200825_H

#include "config.h"
#include "tandem/SeasConfig.h"

#include "mesh/LocalSimplexMesh.h"

namespace tndm {
void solveSEASProblem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg);
// Builds G_ and H_, applies both to a random vector, prints and returns relative error.
// Forces use_hmatrix = true. Returns -1.0 if H2Opus is unavailable.
double validateGFHMatrix(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg);
}

#endif // SEAS_20200825_H
