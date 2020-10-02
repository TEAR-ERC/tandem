#include "PetscBlockVector.h"

#include <cassert>
#include <vector>

namespace tndm {

PetscBlockVector::PetscBlockVector(std::size_t blockSize, std::size_t numLocalElems, MPI_Comm comm)
    : block_size_(blockSize) {
    PetscInt localRows = numLocalElems * blockSize;
    CHKERRTHROW(VecCreate(comm, &x_));
    CHKERRTHROW(VecSetSizes(x_, localRows, PETSC_DECIDE));
    CHKERRTHROW(VecSetFromOptions(x_));
    CHKERRTHROW(VecSetBlockSize(x_, blockSize));
    CHKERRTHROW(VecGetOwnershipRange(x_, &irange_.first, &irange_.second));
}

PetscBlockVector::PetscBlockVector(PetscBlockVector const& prototype)
    : block_size_(prototype.block_size_) {
    VecDuplicate(prototype.vec(), &x_);
    CHKERRTHROW(VecGetOwnershipRange(x_, &irange_.first, &irange_.second));
}

void PetscBlockVector::copy(std::size_t ib, Vector<double>& x) {
    assert(xv_ != nullptr);
    assert(block_size_ == x.size());
    std::size_t i = ib * block_size_;
    assert(irange_.first <= i && i < irange_.second);
    std::size_t offset = i - irange_.first;
    for (std::size_t i = 0; i < block_size_; ++i) {
        x(i) = xv_[offset + i];
    }
}

} // namespace tndm
