#ifndef INTERFACEPETSC_20200727_H
#define INTERFACEPETSC_20200727_H

#include <petscmat.h>
#include <petscvec.h>

#include <vector>

namespace tndm {

class InterfacePetsc {
public:
    InterfacePetsc(PetscInt blockSize, std::size_t numLocalElems, unsigned const* numLocal,
                   unsigned const* numGhost, MPI_Comm comm)
        : blockSize_(blockSize), numLocalElems_(numLocalElems), comm_(comm) {
        d_nnz_aij = std::vector<PetscInt>(localSize());
        o_nnz_aij = std::vector<PetscInt>(localSize());
        for (std::size_t elNo = 0; elNo < numLocalElems_; ++elNo) {
            for (std::size_t b = 0; b < blockSize_; ++b) {
                d_nnz_aij[b + elNo * blockSize_] = numLocal[elNo] * blockSize_;
                o_nnz_aij[b + elNo * blockSize_] = numGhost[elNo] * blockSize_;
            }
        }
        d_nnz_baij = std::vector<PetscInt>(numLocalElems_);
        o_nnz_baij = std::vector<PetscInt>(numLocalElems_);
        for (std::size_t elNo = 0; elNo < numLocalElems_; ++elNo) {
            d_nnz_baij[elNo] = numLocal[elNo];
            o_nnz_baij[elNo] = numGhost[elNo];
        }
    }

    PetscInt localSize() const { return numLocalElems_ * blockSize_; }

    PetscErrorCode createA(Mat* A) {
        CHKERRQ(MatCreate(comm_, A));
        CHKERRQ(MatSetSizes(*A, localSize(), localSize(), PETSC_DETERMINE, PETSC_DETERMINE));
        CHKERRQ(MatSetBlockSize(*A, blockSize_));
        CHKERRQ(MatSetFromOptions(*A));

        // Preallocation
        CHKERRQ(MatSeqAIJSetPreallocation(*A, 0, d_nnz_aij.data()));
        CHKERRQ(MatMPIAIJSetPreallocation(*A, 0, d_nnz_aij.data(), 0, o_nnz_aij.data()));
        CHKERRQ(MatSeqBAIJSetPreallocation(*A, blockSize_, 0, d_nnz_baij.data()));
        CHKERRQ(
            MatMPIBAIJSetPreallocation(*A, blockSize_, 0, d_nnz_baij.data(), 0, o_nnz_baij.data()));

        // Options
        CHKERRQ(MatSetOption(*A, MAT_ROW_ORIENTED, PETSC_FALSE));
        CHKERRQ(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
        return 0;
    }

    PetscErrorCode createb(Vec* b) {
        PetscInt localRows = numLocalElems_ * blockSize_;
        CHKERRQ(VecCreate(comm_, b));
        CHKERRQ(VecSetSizes(*b, localRows, PETSC_DECIDE));
        CHKERRQ(VecSetFromOptions(*b));
        CHKERRQ(VecSetBlockSize(*b, blockSize_));
        return 0;
    }

private:
    PetscInt blockSize_;
    std::size_t numLocalElems_;
    MPI_Comm comm_;
    std::vector<PetscInt> d_nnz_aij, d_nnz_baij, o_nnz_aij, o_nnz_baij;
};

} // namespace tndm

#endif // INTERFACEPETSC_20200727_H
