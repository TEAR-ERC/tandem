#include "PetscVector.h"
#include <petscvec.h>

namespace tndm {

PetscVectorView::PetscVectorView(Vec x) : x_(x) {
    PetscInt bs;
    CHKERRTHROW(VecGetBlockSize(x_, &bs));
    block_size_ = bs;
}

PetscVector::PetscVector(std::size_t blockSize, std::size_t numLocalElems, MPI_Comm comm) {
    PetscInt localRows = numLocalElems * blockSize;
    CHKERRTHROW(VecCreate(comm, &x_));
    CHKERRTHROW(VecSetSizes(x_, localRows, PETSC_DECIDE));
    CHKERRTHROW(VecSetFromOptions(x_));
    CHKERRTHROW(VecSetBlockSize(x_, blockSize));
    block_size_ = blockSize;

    PetscInt local_elems = numLocalElems;
    PetscInt offset = 0;
    MPI_Scan(&local_elems, &offset, 1, MPIU_INT, MPI_SUM, comm);
    offset -= local_elems;

    PetscInt* l2g;
    CHKERRTHROW(PetscMalloc(numLocalElems * sizeof(PetscInt), &l2g));
    for (std::size_t elNo = 0; elNo < numLocalElems; ++elNo) {
        l2g[elNo] = offset + elNo;
    }
    ISLocalToGlobalMapping is_l2g;
    CHKERRTHROW(ISLocalToGlobalMappingCreate(comm, block_size_, numLocalElems, l2g,
                                             PETSC_OWN_POINTER, &is_l2g));
    CHKERRTHROW(VecSetLocalToGlobalMapping(x_, is_l2g));
    CHKERRTHROW(ISLocalToGlobalMappingDestroy(&is_l2g));
}

PetscVector::PetscVector(PetscVector const& prototype) {
    VecDuplicate(prototype.vec(), &x_);
    block_size_ = prototype.block_size_;
}

} // namespace tndm
