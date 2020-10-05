#include "PetscBlockVector.h"

#include <vector>

namespace tndm {

PetscBlockVector::PetscBlockVector(std::size_t blockSize, std::size_t numLocalElems, MPI_Comm comm)
    : block_size_(blockSize) {
    PetscInt localRows = numLocalElems * blockSize;
    CHKERRTHROW(VecCreate(comm, &x_));
    CHKERRTHROW(VecSetSizes(x_, localRows, PETSC_DECIDE));
    CHKERRTHROW(VecSetFromOptions(x_));
    CHKERRTHROW(VecSetBlockSize(x_, blockSize));
}

PetscBlockVector::PetscBlockVector(PetscBlockVector const& prototype)
    : block_size_(prototype.block_size_) {
    VecDuplicate(prototype.vec(), &x_);
}

void PetscBlockVector::copy(const_handle access, std::size_t ib_local, Vector<double>& to) {
    assert(access != nullptr);
    assert(block_size_ == to.size());
    std::size_t i0 = ib_local * block_size_;
    for (std::size_t i = 0; i < block_size_; ++i) {
        to(i) = access[i0 + i];
    }
}

} // namespace tndm
