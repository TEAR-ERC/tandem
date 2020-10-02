#ifndef PETSCBLOCKVECTOR_20201002_H
#define PETSCBLOCKVECTOR_20201002_H

#include "common/PetscUtil.h"

#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <mpi.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include <cstddef>
#include <utility>

namespace tndm {
namespace detail {

class PetscVectorBlock {
public:
    PetscVectorBlock(std::size_t ib, Vector<double>&& values)
        : ib_(ib), values_(std::move(values)) {}

    Vector<double>& values() { return values_; }
    Vector<double> const& values() const { return values_; }
    std::size_t ib() const { return ib_; }

private:
    std::size_t ib_;
    Vector<double> values_;
};

} // namespace detail

class PetscBlockVector {
public:
    PetscBlockVector(std::size_t blockSize, std::size_t numLocalElems, MPI_Comm comm);
    PetscBlockVector(PetscBlockVector const& prototype);
    ~PetscBlockVector() { VecDestroy(&x_); }

    std::size_t scratch_mem_size() const { return sizeof(double) * block_size_; }
    detail::PetscVectorBlock get_block(std::size_t ib, LinearAllocator& scratch) {
        double* buffer = scratch.allocate<double>(block_size_);
        return detail::PetscVectorBlock(ib, Vector<double>(buffer, block_size_));
    }
    void add_block(detail::PetscVectorBlock const& block) {
        PetscInt pib = block.ib();
        VecSetValuesBlocked(x_, 1, &pib, block.values().data(), ADD_VALUES);
    }
    void begin_assembly() {}
    void end_assembly() {
        CHKERRTHROW(VecAssemblyBegin(x_));
        CHKERRTHROW(VecAssemblyEnd(x_));
    }

    void begin_read() { CHKERRTHROW(VecGetArrayRead(x_, &xv_)); }
    void copy(std::size_t ib, Vector<double>& x);
    void end_read() { CHKERRTHROW(VecRestoreArrayRead(x_, &xv_)); }

    void set_zero() { VecZeroEntries(x_); }
    Vec vec() const { return x_; };

private:
    Vec x_;
    std::size_t block_size_;
    PetscScalar const* xv_ = nullptr;
    std::pair<PetscInt, PetscInt> irange_;
};

} // namespace tndm

#endif // PETSCBLOCKVECTOR_20201002_H
