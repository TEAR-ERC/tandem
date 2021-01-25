#include "PetscMatrix.h"

#include <petscsystypes.h>
#include <vector>

namespace tndm {

PetscMatrix::PetscMatrix(std::size_t blockSize, std::size_t numLocalElems, unsigned const* numLocal,
                         unsigned const* numGhost, MPI_Comm comm)
    : block_size_(blockSize) {
    auto localSize = blockSize * numLocalElems;

    CHKERRTHROW(MatCreate(comm, &A_));
    CHKERRTHROW(MatSetSizes(A_, localSize, localSize, PETSC_DETERMINE, PETSC_DETERMINE));
    CHKERRTHROW(MatSetBlockSize(A_, blockSize));
    CHKERRTHROW(MatSetFromOptions(A_));

    // Preallocation
    auto d_nnz_aij = std::vector<PetscInt>(localSize);
    auto o_nnz_aij = std::vector<PetscInt>(localSize);
    for (std::size_t elNo = 0; elNo < numLocalElems; ++elNo) {
        for (std::size_t b = 0; b < blockSize; ++b) {
            d_nnz_aij[b + elNo * blockSize] = numLocal[elNo] * blockSize;
            o_nnz_aij[b + elNo * blockSize] = numGhost[elNo] * blockSize;
        }
    }
    auto d_nnz_baij = std::vector<PetscInt>(numLocalElems);
    auto o_nnz_baij = std::vector<PetscInt>(numLocalElems);
    for (std::size_t elNo = 0; elNo < numLocalElems; ++elNo) {
        d_nnz_baij[elNo] = numLocal[elNo];
        o_nnz_baij[elNo] = numGhost[elNo];
    }

    CHKERRTHROW(MatSeqAIJSetPreallocation(A_, 0, d_nnz_aij.data()));
    CHKERRTHROW(MatMPIAIJSetPreallocation(A_, 0, d_nnz_aij.data(), 0, o_nnz_aij.data()));
    CHKERRTHROW(MatSeqBAIJSetPreallocation(A_, blockSize, 0, d_nnz_baij.data()));
    CHKERRTHROW(
        MatMPIBAIJSetPreallocation(A_, blockSize, 0, d_nnz_baij.data(), 0, o_nnz_baij.data()));

    // Options
    CHKERRTHROW(MatSetOption(A_, MAT_ROW_ORIENTED, PETSC_FALSE));
    CHKERRTHROW(MatSetOption(A_, MAT_SYMMETRIC, PETSC_TRUE));
}

} // namespace tndm
