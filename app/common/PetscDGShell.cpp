#include "PetscDGShell.h"
#include "common/PetscUtil.h"
#include "common/PetscVector.h"
#include <petsc/private/matimpl.h>

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
    CHKERRTHROW(PetscStrallocpy(VECSTANDARD, &A_->defaultvectype)); // seq or mpi
    {
        char val[PETSC_MAX_PATH_LEN];
        PetscBool found = PETSC_FALSE;
        CHKERRTHROW(PetscOptionsGetString(NULL, NULL, "-vec_type", val, sizeof(val), &found));
        if (found) {
            PetscBool isstd, iskok, iscuda, iship;

            CHKERRTHROW(PetscStrcmpAny(val, &isstd, VECSTANDARD, VECSEQ, VECMPI, ""));
            if (isstd) {
                CHKERRTHROW(PetscStrallocpy(VECSTANDARD, &A_->defaultvectype));
            }
            CHKERRTHROW(PetscStrcmpAny(val, &iskok, VECKOKKOS, VECSEQKOKKOS, VECMPIKOKKOS, ""));
            if (iskok) {
                CHKERRTHROW(PetscStrallocpy(VECKOKKOS, &A_->defaultvectype));
            }

            CHKERRTHROW(PetscStrcmpAny(val, &iscuda, VECCUDA, VECSEQCUDA, VECMPICUDA, ""));
            if (iscuda) {
                CHKERRTHROW(PetscStrallocpy(VECCUDA, &A_->defaultvectype));
            }

            CHKERRTHROW(PetscStrcmpAny(val, &iship, VECHIP, VECSEQHIP, VECMPIHIP, ""));
            if (iship) {
                CHKERRTHROW(PetscStrallocpy(VECHIP, &A_->defaultvectype));
            }
        }
    }
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
