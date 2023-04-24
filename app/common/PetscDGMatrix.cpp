#include "PetscDGMatrix.h"
#include "common/PetscUtil.h"

#include "form/DGOperatorTopo.h"
#include "util/Hash.h"

#include <numeric>
#include <petscis.h>
#include <petscmat.h>
#include <petscsystypes.h>

namespace tndm {

PetscDGMatrix::PetscDGMatrix(std::size_t blockSize, DGOperatorTopo const& topo, DGOpSparsityType stype)
    : PetscMatrix(), block_size_(blockSize) {
    const auto numLocalElems = topo.numLocalElements();
    const auto numElems = topo.numElements();
    const auto* gids = topo.gids();
    const auto comm = topo.comm();
    auto localSize = blockSize * numLocalElems;

    CHKERRTHROW(MatCreate(comm, &A_));
    CHKERRTHROW(MatSetSizes(A_, localSize, localSize, PETSC_DETERMINE, PETSC_DETERMINE));
    CHKERRTHROW(MatSetBlockSize(A_, blockSize));
    CHKERRTHROW(MatSetFromOptions(A_));
/*
{
PetscMPIInt rank;
MPI_Comm_rank(comm,&rank);
printf("rank %d: numLocalElems %zu\n",(int)rank,(size_t)numLocalElems);
long int nlmin = (long int)numLocalElems;
long int nlmax = (long int)numLocalElems;
MPI_Allreduce(MPI_IN_PLACE,&nlmin,1,MPI_LONG,MPI_MIN,comm);
MPI_Allreduce(MPI_IN_PLACE,&nlmax,1,MPI_LONG,MPI_MAX,comm);
printf("min el %ld, max el %ld\n",nlmin,nlmax);
}
*/
    PetscPrintf(comm,"DGMatrix blocksize: %zu\n",blockSize);
    // Local to global mapping
    PetscInt* l2g;
    CHKERRTHROW(PetscMalloc(numElems * sizeof(PetscInt), &l2g));
    for (std::size_t elNo = 0; elNo < numElems; ++elNo) {
        l2g[elNo] = gids[elNo];
    }
    ISLocalToGlobalMapping is_l2g;
    CHKERRTHROW(
        ISLocalToGlobalMappingCreate(comm, block_size_, numElems, l2g, PETSC_OWN_POINTER, &is_l2g));
    //--future--//ISLocalToGlobalMappingSetType(is_l2g, ISLOCALTOGLOBALMAPPINGHASH);
    //--future--//ISLocalToGlobalMappingSetFromOptions(is_l2g);
    CHKERRTHROW(MatSetLocalToGlobalMapping(A_, is_l2g, is_l2g));
    CHKERRTHROW(ISLocalToGlobalMappingDestroy(&is_l2g));

    // Options
    CHKERRTHROW(MatSetFromOptions(A_));
    CHKERRTHROW(MatSetUp(A_));

    CHKERRTHROW(MatSetOption(A_, MAT_ROW_ORIENTED, PETSC_FALSE));
    CHKERRTHROW(MatSetOption(A_, MAT_SYMMETRIC, PETSC_TRUE));
    // Preallocation
    MatType type;
    MatGetType(A_, &type);
    switch (fnv1a(type)) {
    case HASH_DEF(MATSEQAIJ):
        preallocate_SeqAIJ(topo, stype);
        fill_SeqAIJ(blockSize, topo, stype);
        break;
    case HASH_DEF(MATMPIAIJ):
        preallocate_MPIAIJ(topo, stype);
        fill_SeqAIJ(blockSize, topo, stype);
        break;
    case HASH_DEF(MATSEQBAIJ):
        preallocate_SeqBAIJ(topo, stype);
        fill_SeqAIJ(blockSize, topo, stype);
        break;
    case HASH_DEF(MATMPIBAIJ):
        preallocate_MPIBAIJ(topo, stype);
        fill_SeqAIJ(blockSize, topo, stype);
        break;
    case HASH_DEF(MATIS):
        preallocate_IS(topo);
        break;
    default:
        break;
    }

    CHKERRTHROW(MatSetOption(A_, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
    //CHKERRTHROW(MatSetOption(A_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
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

void PetscDGMatrix::fill_SeqAIJ(std::size_t bs, DGOperatorTopo const& topo, DGOpSparsityType stype) {
    auto A_size = LinearAllocator<double>::allocation_size(bs * bs, 32);
    auto a_scratch = Scratch<double>(4 * A_size, 32);

    auto scratch_matrix = [&bs](LinearAllocator<double>&scratch) {
        double *buffer = scratch.allocate(bs * bs);
        return Matrix<double>(buffer, bs, bs);
    };


    a_scratch.reset();
    auto Ae = scratch_matrix(a_scratch);
    auto _Ae = Ae.data();
    for (int b=0; b<bs*bs; b++) { _Ae[b] = 0.0; }

    switch (stype) {
      default:
        break;

      case DGOpSparsityType::FULL:
        for (std::size_t elNo=0; elNo<topo.numLocalElements(); ++elNo) {
            add_block(elNo, elNo, Ae);
        }
        for (std::size_t fctNo=0; fctNo<topo.numLocalFacets(); ++fctNo) {
          auto const& info = topo.info(fctNo);
          auto ib0 = info.up[0];
          auto ib1 = info.up[1];
          if (info.up[0] != info.up[1]) {
              if (info.inside[0]) {
                  add_block(ib0, ib0, Ae);
                  add_block(ib0, ib1, Ae);
              }
              if (info.inside[1]) {
                  add_block(ib1, ib0, Ae);
	          add_block(ib1, ib1, Ae);
              }
         } else {
             if (info.inside[0]) {
                 add_block(ib0, ib0, Ae);
             }
           }
        }
        break;

      case DGOpSparsityType::BLOCK_DIAGONAL:
        for (std::size_t elNo=0; elNo<topo.numLocalElements(); ++elNo) {
            add_block(elNo, elNo, Ae);
        }
        break;

      case DGOpSparsityType::DIAGONAL:
      {
        PetscInt m,n,s,e;

        MatGetOwnershipRange(A_,&s,&e);
        MatGetLocalSize(A_,&m,&n);
        for (int k=s; k<e; k++) {
            MatSetValue(A_,k,k,0.0,INSERT_VALUES);
        }
      }
        break;
    }

    CHKERRTHROW(MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY));
    CHKERRTHROW(MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY));

    CHKERRTHROW(MatSetOption(A_, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
    CHKERRTHROW(MatSetOption(A_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    //--future--//MatSetOption(A_,MAT_USE_HASH_TABLE, PETSC_TRUE);
}

void PetscDGMatrix::preallocate_SeqAIJ(DGOperatorTopo const& topo, DGOpSparsityType stype) {
    switch (stype) {
    case DGOpSparsityType::FULL:
    {
      auto d_nnz_aij = nnz_aij(topo.numLocalElements(), topo.numLocalNeighbours());
      CHKERRTHROW(MatSeqAIJSetPreallocation(A_, 0, d_nnz_aij.data()));
    }
      break;

    case DGOpSparsityType::BLOCK_DIAGONAL:
      CHKERRTHROW(MatSeqAIJSetPreallocation(A_, block_size_, NULL));
      break;

    case DGOpSparsityType::DIAGONAL:
      CHKERRTHROW(MatSeqAIJSetPreallocation(A_, 1, NULL));
      break;

    default:
      break;
    }

#if 0
    if (diag_only) {
        CHKERRTHROW(MatSeqAIJSetPreallocation(A_, block_size_, NULL));
        //CHKERRTHROW(MatSeqAIJSetPreallocation(A_, 1, NULL));
    } else {
        auto d_nnz_aij = nnz_aij(topo.numLocalElements(), topo.numLocalNeighbours());
        CHKERRTHROW(MatSeqAIJSetPreallocation(A_, 0, d_nnz_aij.data()));
    }
#endif
}

void PetscDGMatrix::preallocate_MPIAIJ(DGOperatorTopo const& topo, DGOpSparsityType stype) {
  switch (stype) {
    case DGOpSparsityType::FULL:
    {
      auto d_nnz_aij = nnz_aij(topo.numLocalElements(), topo.numLocalNeighbours());
      auto o_nnz_aij = nnz_aij(topo.numLocalElements(), topo.numGhostNeighbours());
      CHKERRTHROW(MatMPIAIJSetPreallocation(A_, 0, d_nnz_aij.data(), 0, o_nnz_aij.data()));
    }
      break;

    case DGOpSparsityType::BLOCK_DIAGONAL:
        CHKERRTHROW(MatMPIAIJSetPreallocation(A_, block_size_, NULL, 0, NULL));
    break;

    case DGOpSparsityType::DIAGONAL:
        CHKERRTHROW(MatMPIAIJSetPreallocation(A_, 1, NULL, 0, NULL));
    break;
  }

#if 0
    if (diag_only) { 
        CHKERRTHROW(MatMPIAIJSetPreallocation(A_, block_size_, NULL, 0, NULL));
    } else {
        auto d_nnz_aij = nnz_aij(topo.numLocalElements(), topo.numLocalNeighbours());
      auto o_nnz_aij = nnz_aij(topo.numLocalElements(), topo.numGhostNeighbours());
      CHKERRTHROW(MatMPIAIJSetPreallocation(A_, 0, d_nnz_aij.data(), 0, o_nnz_aij.data()));
    }
#endif
}

void PetscDGMatrix::preallocate_SeqBAIJ(DGOperatorTopo const& topo, DGOpSparsityType stype) {
  switch (stype) {
    case DGOpSparsityType::FULL:
    {
        auto d_nnz_baij = nnz_baij(topo.numLocalElements(), topo.numLocalNeighbours());
        CHKERRTHROW(MatSeqBAIJSetPreallocation(A_, block_size_, 0, d_nnz_baij.data()));
    }
      break;

    case DGOpSparsityType::BLOCK_DIAGONAL:
        CHKERRTHROW(MatSeqBAIJSetPreallocation(A_, block_size_, 1, NULL));
      break;

    case DGOpSparsityType::DIAGONAL:
        CHKERRTHROW(MatSeqBAIJSetPreallocation(A_, block_size_, 1, NULL));
      break;
  }

#if 0
    if (diag_only) {
        CHKERRTHROW(MatSeqBAIJSetPreallocation(A_, block_size_, 1, NULL));
    } else {
        auto d_nnz_baij = nnz_baij(topo.numLocalElements(), topo.numLocalNeighbours());
        CHKERRTHROW(MatSeqBAIJSetPreallocation(A_, block_size_, 0, d_nnz_baij.data()));
    }
#endif
}

void PetscDGMatrix::preallocate_MPIBAIJ(DGOperatorTopo const& topo, DGOpSparsityType stype) {
  switch (stype) {
    case DGOpSparsityType::FULL:
    {
        auto d_nnz_baij = nnz_baij(topo.numLocalElements(), topo.numLocalNeighbours());
        auto o_nnz_baij = nnz_baij(topo.numLocalElements(), topo.numGhostNeighbours());
        CHKERRTHROW(
          MatMPIBAIJSetPreallocation(A_, block_size_, 0, d_nnz_baij.data(), 0, o_nnz_baij.data())
                   );
    }
      break;

    case DGOpSparsityType::BLOCK_DIAGONAL:
        CHKERRTHROW(MatMPIBAIJSetPreallocation(A_, block_size_, 1, NULL, 0, NULL));
    break;

    case DGOpSparsityType::DIAGONAL:
        CHKERRTHROW(MatMPIBAIJSetPreallocation(A_, block_size_, 1, NULL, 0, NULL));
    break;
  }

#if 0
    if (diag_only) {
        CHKERRTHROW(MatMPIBAIJSetPreallocation(A_,block_size_, 1, NULL, 0, NULL));
    } else {
        auto d_nnz_baij = nnz_baij(topo.numLocalElements(), topo.numLocalNeighbours());
        auto o_nnz_baij = nnz_baij(topo.numLocalElements(), topo.numGhostNeighbours());
        CHKERRTHROW(
          MatMPIBAIJSetPreallocation(A_, block_size_, 0, d_nnz_baij.data(), 0, o_nnz_baij.data())
                   );
    }
#endif

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
