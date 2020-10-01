#ifndef SEAS_20200825_H
#define SEAS_20200825_H

#include "config.h"
#include "tandem/Config.h"

#include "mesh/LocalSimplexMesh.h"

namespace tndm {
void solveSEASProblem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg);
}

#endif // SEAS_20200825_H
