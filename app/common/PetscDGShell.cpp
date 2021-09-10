#include "PetscDGShell.h"
#include "common/PetscUtil.h"
#include "common/PetscVector.h"

namespace tndm {

PetscDGShell::PetscDGShell(AbstractDGOperator<DomainDimension>& dgop) {
    const auto blockSize = dgop.block_size();
    const auto localSize = blockSize * dgop.num_local_elements();
    const auto comm = dgop.topo().comm();
    CHKERRTHROW(MatCreate(comm, &A_));
    CHKERRTHROW(MatSetSizes(A_, localSize, localSize, PETSC_DETERMINE, PETSC_DETERMINE));
    CHKERRTHROW(MatSetBlockSize(A_, blockSize));
    CHKERRTHROW(MatSetType(A_, MATSHELL));
    CHKERRTHROW(MatSetUp(A_));

    CHKERRTHROW(MatShellSetContext(A_, static_cast<void*>(&dgop)));
    CHKERRTHROW(MatShellSetOperation(A_, MATOP_MULT, (void (*)(void))apply));
}

PetscDGShell::~PetscDGShell() { MatDestroy(&A_); }

PetscErrorCode PetscDGShell::apply(Mat A, Vec x, Vec y) {
    AbstractDGOperator<DomainDimension>* dgop;
    MatShellGetContext(A, &dgop);
    const auto xv = PetscVectorView(x);
    auto yv = PetscVectorView(y);
    dgop->apply(xv, yv);
    return 0;
}

} // namespace tndm
