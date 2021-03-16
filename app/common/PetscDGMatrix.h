#ifndef PETSCDGMATRIX_20210208_H
#define PETSCDGMATRIX_20210208_H

#include "common/PetscMatrix.h"

#include "form/DGOperatorTopo.h"

#include <petscsystypes.h>

#include <cstddef>
#include <vector>

namespace tndm {

class PetscDGMatrix : public PetscMatrix {
public:
    PetscDGMatrix(std::size_t blockSize, DGOperatorTopo const& topo);

private:
    std::vector<PetscInt> nnz_aij(std::size_t numLocalElems, unsigned const* nnz);
    std::vector<PetscInt> nnz_baij(std::size_t numLocalElems, unsigned const* nnz);
    void preallocate_SeqAIJ(DGOperatorTopo const& topo);
    void preallocate_MPIAIJ(DGOperatorTopo const& topo);
    void preallocate_SeqBAIJ(DGOperatorTopo const& topo);
    void preallocate_MPIBAIJ(DGOperatorTopo const& topo);
    void preallocate_IS(DGOperatorTopo const& topo);

    std::size_t block_size_;
};

} // namespace tndm

#endif // PETSCDGMATRIX_20210208_H
