#include "PetscInterplMatrix.h"
#include "common/PetscUtil.h"
#include <petscmat.h>

namespace tndm {

PetscInterplMatrix::PetscInterplMatrix(std::size_t rowBlockSize, std::size_t columnBlockSize,
                                       DGOperatorTopo const& topo)
    : PetscMatrix() {
    const auto numLocalElems = topo.numLocalElements();
    const auto* gids = topo.gids();
    const auto comm = topo.comm();
    auto localRows = rowBlockSize * numLocalElems;
    auto localCols = columnBlockSize * numLocalElems;

    CHKERRTHROW(MatCreate(comm, &A_));
    CHKERRTHROW(MatSetSizes(A_, localRows, localCols, PETSC_DETERMINE, PETSC_DETERMINE));
    CHKERRTHROW(MatSetBlockSizes(A_, rowBlockSize, columnBlockSize));
    CHKERRTHROW(MatSetType(A_, MATAIJ));
    CHKERRTHROW(MatSetFromOptions(A_));

    // Local to global mapping
    PetscInt* l2g;
    CHKERRTHROW(PetscMalloc(numLocalElems * sizeof(PetscInt), &l2g));
    for (std::size_t elNo = 0; elNo < numLocalElems; ++elNo) {
        l2g[elNo] = gids[elNo];
    }
    ISLocalToGlobalMapping is_l2g_row, is_l2g_col;
    CHKERRTHROW(ISLocalToGlobalMappingCreate(comm, rowBlockSize, numLocalElems, l2g,
                                             PETSC_COPY_VALUES, &is_l2g_row));
    CHKERRTHROW(ISLocalToGlobalMappingCreate(comm, columnBlockSize, numLocalElems, l2g,
                                             PETSC_OWN_POINTER, &is_l2g_col));
    CHKERRTHROW(MatSetLocalToGlobalMapping(A_, is_l2g_row, is_l2g_col));
    CHKERRTHROW(ISLocalToGlobalMappingDestroy(&is_l2g_row));
    CHKERRTHROW(ISLocalToGlobalMappingDestroy(&is_l2g_col));

    // Preallocation
    CHKERRTHROW(MatSeqAIJSetPreallocation(A_, columnBlockSize, nullptr));
    CHKERRTHROW(MatMPIAIJSetPreallocation(A_, columnBlockSize, nullptr, 0, nullptr));

    // Options
    CHKERRTHROW(MatSetOption(A_, MAT_ROW_ORIENTED, PETSC_FALSE));
}

} // namespace tndm
