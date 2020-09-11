#include "PetscSolver.h"
#include "common/PetscUtil.h"

#include <vector>

namespace tndm {

void PetscSolver::create_mat(std::size_t blockSize, std::size_t numLocalElems,
                             unsigned const* numLocal, unsigned const* numGhost, bool reuse,
                             MPI_Comm comm) {
    numLocalElems_ = numLocalElems;
    blockSize_ = blockSize;
    auto localSize = blockSize * numLocalElems;

    if (A_ != nullptr) {
        if (reuse) {
            MatZeroEntries(A_);
            return;
        } else {
            MatDestroy(&A_);
        }
    }

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

void PetscSolver::create_vec(std::size_t blockSize, std::size_t numLocalElems, bool reuse,
                             MPI_Comm comm) {
    if (b_ != nullptr) {
        if (reuse) {
            VecZeroEntries(b_);
            return;
        } else {
            VecDestroy(&b_);
        }
    }

    PetscInt localRows = numLocalElems * blockSize;
    CHKERRTHROW(VecCreate(comm, &b_));
    CHKERRTHROW(VecSetSizes(b_, localRows, PETSC_DECIDE));
    CHKERRTHROW(VecSetFromOptions(b_));
    CHKERRTHROW(VecSetBlockSize(b_, blockSize));
}

void PetscSolver::setup_solve() {
    if (ksp_ != nullptr) {
        KSPDestroy(&ksp_);
    }
    CHKERRTHROW(KSPCreate(PETSC_COMM_WORLD, &ksp_));
    CHKERRTHROW(KSPSetType(ksp_, KSPCG));
    CHKERRTHROW(KSPSetOperators(ksp_, A_, A_));
    CHKERRTHROW(KSPSetTolerances(ksp_, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    CHKERRTHROW(KSPSetFromOptions(ksp_));
}

} // namespace tndm
