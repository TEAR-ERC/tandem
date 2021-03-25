#ifndef PETSCVECTOR_20210125_H
#define PETSCVECTOR_20210125_H

#include "common/PetscUtil.h"

#include "tensor/Tensor.h"

#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace tndm {

class PetscVectorView {
public:
    using handle = PetscScalar*;
    using const_handle = PetscScalar const*;
    using value_type = PetscScalar;

    PetscVectorView() {}
    PetscVectorView(Vec x);

    std::size_t block_size() const { return block_size_; }

    void add_block(std::size_t ib_local, Vector<double> const& values) {
        PetscInt pib = ib_local;
        VecSetValuesBlockedLocal(x_, 1, &pib, values.data(), ADD_VALUES);
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
    const_handle begin_access_readonly() const {
        const_handle xv;
        CHKERRTHROW(VecGetArrayRead(x_, &xv));
        return xv;
    }
    void end_access_readonly(const_handle xv) const { CHKERRTHROW(VecRestoreArrayRead(x_, &xv)); }

    void copy(const_handle access, std::size_t ib_local, Vector<double>& to) const;
    auto get_block(handle access, std::size_t ib_local) {
        static_assert(std::is_same_v<PetscScalar, double>, "PetscScalar must be double");
        assert(access != nullptr);
        return Vector<double>(&access[ib_local * block_size_], block_size_);
    }
    auto get_block(const_handle access, std::size_t ib_local) const {
        static_assert(std::is_same_v<PetscScalar, double>, "PetscScalar must be double");
        assert(access != nullptr);
        return Vector<double const>(&access[ib_local * block_size_], block_size_);
    }

    void set_zero() { VecZeroEntries(x_); }
    Vec vec() const { return x_; };

protected:
    Vec x_ = nullptr;
    std::size_t block_size_ = 0;
};

class PetscVector : public PetscVectorView {
public:
    using handle = PetscScalar*;
    using const_handle = PetscScalar const*;

    PetscVector(std::size_t blockSize, std::size_t numLocalElems, MPI_Comm comm);
    PetscVector(PetscVector const& prototype);
    ~PetscVector() { VecDestroy(&x_); }
};

} // namespace tndm

#endif // PETSCVECTOR_20210125_H
