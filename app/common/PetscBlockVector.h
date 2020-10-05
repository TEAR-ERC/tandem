#ifndef PETSCBLOCKVECTOR_20201002_H
#define PETSCBLOCKVECTOR_20201002_H

#include "common/PetscUtil.h"

#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <mpi.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tndm {

class PetscBlockVector {
public:
    using handle = PetscScalar*;
    using const_handle = PetscScalar const*;

    PetscBlockVector(std::size_t blockSize, std::size_t numLocalElems, MPI_Comm comm);
    PetscBlockVector(PetscBlockVector const& prototype);
    ~PetscBlockVector() { VecDestroy(&x_); }

    std::size_t scratch_mem_size() const { return sizeof(double) * block_size_; }
    void add_block(std::size_t ib_global, Vector<double> const& values) {
        PetscInt pib = ib_global;
        VecSetValuesBlocked(x_, 1, &pib, values.data(), ADD_VALUES);
    }
    void begin_assembly() {}
    void end_assembly() {
        CHKERRTHROW(VecAssemblyBegin(x_));
        CHKERRTHROW(VecAssemblyEnd(x_));
    }

    handle begin_access() {
        handle xv;
        CHKERRTHROW(VecGetArray(x_, &xv));
        return xv;
    }
    void end_access(handle xv) { CHKERRTHROW(VecRestoreArray(x_, &xv)); }
    const_handle begin_access_readonly() {
        const_handle xv;
        CHKERRTHROW(VecGetArrayRead(x_, &xv));
        return xv;
    }
    void end_access_readonly(const_handle xv) { CHKERRTHROW(VecRestoreArrayRead(x_, &xv)); }

    void copy(const_handle access, std::size_t ib_local, Vector<double>& to);
    auto get_block(handle access, std::size_t ib_local) {
        static_assert(std::is_same_v<PetscScalar, double>, "PetscScalar must be double");
        assert(access != nullptr);
        return Vector<double>(&access[ib_local * block_size_], block_size_);
    }
    auto get_block(const_handle access, std::size_t ib_local) {
        static_assert(std::is_same_v<PetscScalar, double>, "PetscScalar must be double");
        assert(access != nullptr);
        return Vector<double const>(&access[ib_local * block_size_], block_size_);
    }

    void set_zero() { VecZeroEntries(x_); }
    Vec vec() const { return x_; };

private:
    Vec x_;
    std::size_t block_size_;
};

} // namespace tndm

#endif // PETSCBLOCKVECTOR_20201002_H
