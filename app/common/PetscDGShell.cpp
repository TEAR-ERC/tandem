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
    {
        char val[PETSC_MAX_PATH_LEN];
        PetscBool found = PETSC_FALSE;
        CHKERRTHROW(PetscOptionsGetString(NULL, NULL, "-vec_type", val, sizeof(val), &found));

        if (!found) {
            CHKERRTHROW(PetscStrallocpy(VECSTANDARD, &A_->defaultvectype)); // seq or mpi
        } else {
            const char *vecTypes[] = {VECSTANDARD, VECKOKKOS, VECCUDA, VECHIP};
            const char *vecSeqTypes[] = {VECSEQ, VECSEQKOKKOS, VECSEQCUDA, VECSEQHIP};
            const char *vecMPItypes[] = {VECMPI, VECMPIKOKKOS, VECMPICUDA, VECMPIHIP};
            PetscBool match;

            // Check for Kokkos, CUDA, or HIP vector types only
            for (int i = 0; i < 4; ++i) {
                CHKERRTHROW(PetscStrcmpAny(val, &match, vecTypes[i], vecSeqTypes[i], vecMPItypes[i], ""));
                if (match) {
                    CHKERRTHROW(PetscStrallocpy(vecTypes[i], &A_->defaultvectype));
                    break;
                }
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
