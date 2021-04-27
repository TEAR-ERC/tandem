#ifndef PETSCDGSHELL_20210302_H
#define PETSCDGSHELL_20210302_H

#include "common/PetscUtil.h"
#include "common/PetscVector.h"

#include "form/DGOperator.h"

#include <petscmat.h>

namespace tndm {

class PetscDGShell {
public:
    template <typename DGOp> PetscDGShell(DGOp& dgop) {
        const auto blockSize = dgop.block_size();
        const auto localSize = blockSize * dgop.numLocalElements();
        const auto comm = dgop.topo().comm();
        CHKERRTHROW(MatCreate(comm, &A_));
        CHKERRTHROW(MatSetSizes(A_, localSize, localSize, PETSC_DETERMINE, PETSC_DETERMINE));
        CHKERRTHROW(MatSetBlockSize(A_, blockSize));
        CHKERRTHROW(MatSetType(A_, MATSHELL));
        CHKERRTHROW(MatSetUp(A_));

        CHKERRTHROW(MatShellSetContext(A_, static_cast<void*>(&dgop)));
        CHKERRTHROW(MatShellSetOperation(A_, MATOP_MULT, (void (*)(void))apply<DGOp>));
    }
    ~PetscDGShell() { MatDestroy(&A_); }

    Mat mat() const { return A_; };

private:
    template <typename DGOp> static void apply(Mat A, Vec x, Vec y) {
        DGOp* dgop;
        MatShellGetContext(A, &dgop);
        const auto xv = PetscVectorView(x);
        auto yv = PetscVectorView(y);
        dgop->apply(xv, yv);
    }

    Mat A_;
};

} // namespace tndm

#endif // PETSCDGSHELL_20210302_H
