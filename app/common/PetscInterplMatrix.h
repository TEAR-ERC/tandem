#ifndef PETSCINTERPLMATRIX_20210315_H
#define PETSCINTERPLMATRIX_20210315_H

#include "common/PetscMatrix.h"

#include "form/DGOperatorTopo.h"

#include <cstddef>
#include <mpi.h>

namespace tndm {

class PetscInterplMatrix : public PetscMatrix {
public:
    PetscInterplMatrix(std::size_t rowBlockSize, std::size_t columnBlockSize,
                       DGOperatorTopo const& topo);
};

} // namespace tndm

#endif // PETSCINTERPLMATRIX_20210315_H
