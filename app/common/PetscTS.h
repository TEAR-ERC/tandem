#ifndef PETSCTS_20201001_H
#define PETSCTS_20201001_H

#include "common/PetscUtil.h"
#include "tensor/Tensor.h"

#include <petscsys.h>
#include <petscsystypes.h>
#include <petscts.h>
#include <petscvec.h>

#include <cstddef>

namespace tndm {

class PetscTS {
public:
    ~PetscTS() {
        if (x_ != nullptr) {
            VecDestroy(&x_);
        }
        if (ts_ != nullptr) {
            TSDestroy(&ts_);
        }
    }

    void create_state(std::size_t blockSize, std::size_t numLocalElems, bool reuse, MPI_Comm comm);

    void begin_state() { CHKERRTHROW(VecGetArray(x_, &xv_)); }
    Vector<double> get_state(std::size_t ib) {
        PetscInt blockSize;
        VecGetBlockSize(x_, &blockSize);
        return Vector<double>(xv_ + ib * blockSize, blockSize);
    }
    void end_state() {
        CHKERRTHROW(VecRestoreArray(x_, &xv_));
        CHKERRTHROW(VecAssemblyBegin(x_));
        CHKERRTHROW(VecAssemblyEnd(x_));
    }

private:
    Vec x_ = nullptr;
    TS ts_ = nullptr;
    std::size_t numLocalElems_ = 0;

    PetscScalar* xv_;
};

} // namespace tndm

#endif // PETSCTS_20201001_H
