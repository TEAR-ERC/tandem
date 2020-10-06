#include "PetscBlockVector.h"

#include <vector>

namespace tndm {

PetscBlockVectorView::PetscBlockVectorView(Vec x) : x_(x) {
    PetscInt bs;
    CHKERRTHROW(VecGetBlockSize(x_, &bs));
    block_size_ = bs;
}

void PetscBlockVectorView::copy(const_handle access, std::size_t ib_local,
                                Vector<double>& to) const {
    assert(access != nullptr);
    assert(block_size_ == to.size());
    std::size_t i0 = ib_local * block_size_;
    for (std::size_t i = 0; i < block_size_; ++i) {
        to(i) = access[i0 + i];
    }
}

PetscBlockVector::PetscBlockVector(std::size_t blockSize, std::size_t numLocalElems,
                                   MPI_Comm comm) {
    PetscInt localRows = numLocalElems * blockSize;
    CHKERRTHROW(VecCreate(comm, &x_));
    CHKERRTHROW(VecSetSizes(x_, localRows, PETSC_DECIDE));
    CHKERRTHROW(VecSetFromOptions(x_));
    CHKERRTHROW(VecSetBlockSize(x_, blockSize));
    block_size_ = blockSize;
}

PetscBlockVector::PetscBlockVector(PetscBlockVector const& prototype) {
    VecDuplicate(prototype.vec(), &x_);
    block_size_ = prototype.block_size_;
}

} // namespace tndm
