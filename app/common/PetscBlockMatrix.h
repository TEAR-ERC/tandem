#ifndef PETSCBLOCKMATRIX_20201002_H
#define PETSCBLOCKMATRIX_20201002_H

#include "common/PetscUtil.h"

#include "tensor/Tensor.h"

#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>

#include <cstddef>

namespace tndm {

class PetscBlockMatrix {
public:
    PetscBlockMatrix(std::size_t blockSize, std::size_t numLocalElems, unsigned const* numLocal,
                     unsigned const* numGhost, MPI_Comm comm);
    ~PetscBlockMatrix() { MatDestroy(&A_); }

    void add_block(std::size_t ib_global, std::size_t jb_global, Matrix<double> const& values) {
        PetscInt pib = ib_global, pjb = jb_global;
        MatSetValuesBlocked(A_, 1, &pib, 1, &pjb, values.data(), ADD_VALUES);
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
