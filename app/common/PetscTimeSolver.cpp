#include "PetscTimeSolver.h"

namespace tndm {

void PetscTimeSolver::create_state(std::size_t blockSize, std::size_t numLocalElems, bool reuse,
                                   MPI_Comm comm) {
    if (x_ != nullptr) {
        if (reuse) {
            VecZeroEntries(x_);
            return;
        } else {
            VecDestroy(&x_);
        }
    }

    PetscInt localRows = numLocalElems * blockSize;
    CHKERRTHROW(VecCreate(comm, &x_));
    CHKERRTHROW(VecSetSizes(x_, localRows, PETSC_DECIDE));
    CHKERRTHROW(VecSetFromOptions(x_));
    CHKERRTHROW(VecSetBlockSize(x_, blockSize));
}

} // namespace tndm
