#ifndef PETSCDGMATRIX_20210208_H
#define PETSCDGMATRIX_20210208_H

#include "common/PetscMatrix.h"

#include "form/DGOperatorTopo.h"

//#include "util/LinearAllocator.h"
#include "util/Scratch.h"

#include <petscsystypes.h>

#include <cstddef>
#include <vector>

namespace tndm {

enum class DGOpSparsityType { FULL=0, BLOCK_DIAGONAL=1, FIELD_DIAGONAL=2, DIAGONAL=3 };

class PetscDGMatrix : public PetscMatrix {
public:
    PetscDGMatrix(std::size_t blockSize, DGOperatorTopo const& topo, DGOpSparsityType stype);

private:
    std::vector<PetscInt> nnz_aij(std::size_t numLocalElems, unsigned const* nnz);
    std::vector<PetscInt> nnz_baij(std::size_t numLocalElems, unsigned const* nnz);
    void preallocate_SeqAIJ(DGOperatorTopo const& topo,DGOpSparsityType stype);
    void preallocate_MPIAIJ(DGOperatorTopo const& topo,DGOpSparsityType stype);
    void preallocate_SeqBAIJ(DGOperatorTopo const& topo,DGOpSparsityType stype);
    void preallocate_MPIBAIJ(DGOperatorTopo const& topo,DGOpSparsityType stype);
    void preallocate_IS(DGOperatorTopo const& topo);

    void fill_SeqAIJ(std::size_t bs, DGOperatorTopo const& topo,DGOpSparsityType stype);

    //Scratch<double> scratch_;
    std::size_t block_size_;
};

} // namespace tndm

#endif // PETSCDGMATRIX_20210208_H
