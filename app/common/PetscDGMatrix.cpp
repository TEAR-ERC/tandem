#include "PetscDGMatrix.h"
#include "common/PetscUtil.h"

#include "form/DGOperatorTopo.h"
#include "util/Hash.h"

#include <numeric>
#include <petscis.h>
#include <petscmat.h>
#include <petscsystypes.h>

namespace tndm {

PetscDGMatrix::PetscDGMatrix(std::size_t blockSize, DGOperatorTopo const& topo)
    : block_size_(blockSize) {
    const auto numLocalElems = topo.numLocalElements();
    const auto numElems = topo.numElements();
    const auto* gids = topo.gids();
    const auto comm = topo.comm();
    auto localSize = blockSize * numLocalElems;

    CHKERRTHROW(MatCreate(comm, &A_));
    CHKERRTHROW(MatSetSizes(A_, localSize, localSize, PETSC_DETERMINE, PETSC_DETERMINE));
    CHKERRTHROW(MatSetBlockSize(A_, blockSize));
    CHKERRTHROW(MatSetFromOptions(A_));

    // Local to global mapping
    PetscInt* l2g;
    CHKERRTHROW(PetscMalloc(numElems * sizeof(PetscInt), &l2g));
    for (std::size_t elNo = 0; elNo < numElems; ++elNo) {
        l2g[elNo] = gids[elNo];
    }
    ISLocalToGlobalMapping is_l2g;
    CHKERRTHROW(
        ISLocalToGlobalMappingCreate(comm, block_size_, numElems, l2g, PETSC_OWN_POINTER, &is_l2g));
    CHKERRTHROW(MatSetLocalToGlobalMapping(A_, is_l2g, is_l2g));
    CHKERRTHROW(ISLocalToGlobalMappingDestroy(&is_l2g));

    // Preallocation
    MatType type;
    MatGetType(A_, &type);
    switch (fnv1a(type)) {
    case HASH_DEF(MATSEQAIJ):
        preallocate_SeqAIJ(topo);
        break;
    case HASH_DEF(MATMPIAIJ):
        preallocate_MPIAIJ(topo);
        break;
    case HASH_DEF(MATSEQBAIJ):
        preallocate_SeqBAIJ(topo);
        break;
    case HASH_DEF(MATMPIBAIJ):
        preallocate_MPIBAIJ(topo);
        break;
    case HASH_DEF(MATIS):
        preallocate_IS(topo);
        break;
    default:
        break;
    }

    // Options
    CHKERRTHROW(MatSetOption(A_, MAT_ROW_ORIENTED, PETSC_FALSE));
    CHKERRTHROW(MatSetOption(A_, MAT_SYMMETRIC, PETSC_TRUE));
}

std::vector<PetscInt> PetscDGMatrix::nnz_aij(std::size_t numLocalElems, unsigned const* nnz) {
    auto nnz_new = std::vector<PetscInt>(block_size_ * numLocalElems);
    for (std::size_t elNo = 0; elNo < numLocalElems; ++elNo) {
        for (std::size_t b = 0; b < block_size_; ++b) {
            nnz_new[b + elNo * block_size_] = nnz[elNo] * block_size_;
        }
    }
    return nnz_new;
}

std::vector<PetscInt> PetscDGMatrix::nnz_baij(std::size_t numLocalElems, unsigned const* nnz) {
    auto nnz_new = std::vector<PetscInt>(numLocalElems);
    for (std::size_t elNo = 0; elNo < numLocalElems; ++elNo) {
        nnz_new[elNo] = nnz[elNo];
    }
    return nnz_new;
}

void PetscDGMatrix::preallocate_SeqAIJ(DGOperatorTopo const& topo) {
    auto d_nnz_aij = nnz_aij(topo.numLocalElements(), topo.numLocalNeighbours());
    CHKERRTHROW(MatSeqAIJSetPreallocation(A_, 0, d_nnz_aij.data()));
}

void PetscDGMatrix::preallocate_MPIAIJ(DGOperatorTopo const& topo) {
    auto d_nnz_aij = nnz_aij(topo.numLocalElements(), topo.numLocalNeighbours());
    auto o_nnz_aij = nnz_aij(topo.numLocalElements(), topo.numGhostNeighbours());
    CHKERRTHROW(MatMPIAIJSetPreallocation(A_, 0, d_nnz_aij.data(), 0, o_nnz_aij.data()));
}

void PetscDGMatrix::preallocate_SeqBAIJ(DGOperatorTopo const& topo) {
    auto d_nnz_baij = nnz_baij(topo.numLocalElements(), topo.numLocalNeighbours());
    CHKERRTHROW(MatSeqBAIJSetPreallocation(A_, block_size_, 0, d_nnz_baij.data()));
}

void PetscDGMatrix::preallocate_MPIBAIJ(DGOperatorTopo const& topo) {
    auto d_nnz_baij = nnz_baij(topo.numLocalElements(), topo.numLocalNeighbours());
    auto o_nnz_baij = nnz_baij(topo.numLocalElements(), topo.numGhostNeighbours());
    CHKERRTHROW(
        MatMPIBAIJSetPreallocation(A_, block_size_, 0, d_nnz_baij.data(), 0, o_nnz_baij.data()));
}

void PetscDGMatrix::preallocate_IS(DGOperatorTopo const& topo) {
    const auto numLocalElems = topo.numLocalElements();
    const auto numElems = topo.numElements();
    const auto* numLocal = topo.numLocalNeighbours();
    const auto* numGhost = topo.numGhostNeighbours();
    auto nnz = std::vector<PetscInt>(block_size_ * numElems, 0);
    for (std::size_t elNo = 0; elNo < numLocalElems; ++elNo) {
        auto blocks = numLocal[elNo] + numGhost[elNo];
        for (std::size_t b = 0; b < block_size_; ++b) {
            nnz[b + elNo * block_size_] = blocks * block_size_;
        }
    }
    for (std::size_t i = numLocalElems * block_size_; i < numElems * block_size_; ++i) {
        nnz[i] += 1;
    }
    Mat lA;
    CHKERRTHROW(MatISGetLocalMat(A_, &lA));
    CHKERRTHROW(MatSeqAIJSetPreallocation(lA, 0, nnz.data()));
    CHKERRTHROW(MatISRestoreLocalMat(A_, &lA));
}

} // namespace tndm
