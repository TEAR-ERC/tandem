#ifndef PETSCVECTOR_20210125_H
#define PETSCVECTOR_20210125_H

#include "common/PetscUtil.h"

#include "interface/BlockVector.h"
#include "tensor/Tensor.h"

#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace tndm {

class PetscVectorView : public BlockVector {
public:
    PetscVectorView() {}
    PetscVectorView(Vec x);

    std::size_t block_size() const { return block_size_; }

    void add_block(std::size_t ib_local, Vector<double> const& values) {
        PetscInt pib = ib_local;
        VecSetValuesBlockedLocal(x_, 1, &pib, values.data(), ADD_VALUES);
    }
    void add_block(std::size_t ib_local, Vector<const double> const& values) {
        PetscInt pib = ib_local;
        VecSetValuesBlockedLocal(x_, 1, &pib, values.data(), ADD_VALUES);
    }
    void insert_block(std::size_t ib_local, Vector<double> const& values) {
        PetscInt pib = ib_local;
        VecSetValuesBlockedLocal(x_, 1, &pib, values.data(), INSERT_VALUES);
    }
    void insert_block(std::size_t ib_local, Vector<const double> const& values) {
        PetscInt pib = ib_local;
        VecSetValuesBlockedLocal(x_, 1, &pib, values.data(), INSERT_VALUES);
    }
    void begin_assembly() {}
    void end_assembly() {
        CHKERRTHROW(VecAssemblyBegin(x_));
        CHKERRTHROW(VecAssemblyEnd(x_));
    }

    Matrix<double> begin_access() {
        static_assert(std::is_same_v<PetscScalar, double>, "PetscScalar must be double");
        PetscScalar* xv;
        PetscInt size;
        CHKERRTHROW(VecGetArray(x_, &xv));
        CHKERRTHROW(VecGetLocalSize(x_, &size));
        assert(size % block_size_ == 0);
        return Matrix<double>(xv, block_size_, size / block_size_);
    }
    void end_access(Matrix<double>& data) {
        PetscScalar* xv = data.data();
        CHKERRTHROW(VecRestoreArray(x_, &xv));
    }
    Matrix<const double> begin_access_readonly() const {
        static_assert(std::is_same_v<PetscScalar, double>, "PetscScalar must be double");
        PetscScalar const* xv;
        PetscInt size;
        CHKERRTHROW(VecGetArrayRead(x_, &xv));
        CHKERRTHROW(VecGetLocalSize(x_, &size));
        assert(size % block_size_ == 0);
        return Matrix<const double>(xv, block_size_, size / block_size_);
    }
    void end_access_readonly(Matrix<const double>& data) const {
        PetscScalar const* xv = data.data();
        CHKERRTHROW(VecRestoreArrayRead(x_, &xv));
    }

    void set_zero() { VecZeroEntries(x_); }
    Vec vec() const { return x_; };

protected:
    Vec x_ = nullptr;
    std::size_t block_size_ = 0;
};

class PetscVector : public PetscVectorView {
public:
    PetscVector(std::size_t blockSize, std::size_t numLocalElems, MPI_Comm comm);
    PetscVector(PetscVector const& prototype);
    ~PetscVector() { VecDestroy(&x_); }
};

} // namespace tndm

#endif // PETSCVECTOR_20210125_H
