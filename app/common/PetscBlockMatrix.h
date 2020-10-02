#ifndef PETSCBLOCKMATRIX_20201002_H
#define PETSCBLOCKMATRIX_20201002_H

#include "common/PetscUtil.h"

#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <mpi.h>
#include <petscmat.h>
#include <petscsystypes.h>

#include <cstddef>
#include <utility>

namespace tndm {

namespace detail {
class PetscMatrixBlock {
public:
    PetscMatrixBlock(std::size_t ib, std::size_t jb, Matrix<double>&& values)
        : ib_(ib), jb_(jb), values_(std::move(values)) {}

    Matrix<double>& values() { return values_; }
    Matrix<double> const& values() const { return values_; }
    std::size_t ib() const { return ib_; }
    std::size_t jb() const { return jb_; }

private:
    std::size_t ib_, jb_;
    Matrix<double> values_;
};
} // namespace detail

class PetscBlockMatrix {
public:
    PetscBlockMatrix(std::size_t blockSize, std::size_t numLocalElems, unsigned const* numLocal,
                     unsigned const* numGhost, MPI_Comm comm);
    ~PetscBlockMatrix() { MatDestroy(&A_); }

    std::size_t scratch_mem_size() const { return sizeof(double) * block_size_ * block_size_; }
    detail::PetscMatrixBlock get_block(std::size_t ib, std::size_t jb, LinearAllocator& scratch) {
        double* buffer = scratch.allocate<double>(block_size_ * block_size_);
        return detail::PetscMatrixBlock(ib, jb, Matrix<double>(buffer, block_size_, block_size_));
    }
    void add_block(detail::PetscMatrixBlock const& block) {
        PetscInt pib = block.ib(), pjb = block.jb();
        MatSetValuesBlocked(A_, 1, &pib, 1, &pjb, block.values().data(), ADD_VALUES);
    }
    void begin_assembly() {}
    void end_assembly() {
        CHKERRTHROW(MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY));
        CHKERRTHROW(MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY));
    }

    void set_zero() { MatZeroEntries(A_); }
    Mat mat() const { return A_; };

private:
    Mat A_;
    std::size_t block_size_;
};

} // namespace tndm

#endif // PETSCBLOCKMATRIX_20201002_H
