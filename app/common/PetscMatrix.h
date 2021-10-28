#ifndef PETSCMATRIX_20210315_H
#define PETSCMATRIX_20210315_H

#include "common/PetscUtil.h"

#include "interface/BlockMatrix.h"
#include "tensor/Tensor.h"

#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>

#include <cstddef>

namespace tndm {

class PetscMatrix : public BlockMatrix {
public:
    ~PetscMatrix() { MatDestroy(&A_); }

    void add_block(std::size_t ib_local, std::size_t jb_local,
                   Matrix<double> const& values) override {
        PetscInt pib = ib_local, pjb = jb_local;
        MatSetValuesBlockedLocal(A_, 1, &pib, 1, &pjb, values.data(), ADD_VALUES);
    }
    void begin_assembly() override {}
    void end_assembly() override {
        CHKERRTHROW(MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY));
        CHKERRTHROW(MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY));
    }

    void set_zero() override { MatZeroEntries(A_); }
    Mat mat() const { return A_; };

protected:
    Mat A_;
};

} // namespace tndm

#endif // PETSCMATRIX_20210315_H
