#include "PetscLinearSolver.h"
#include "common/PetscUtil.h"
#include <petscpc.h>

namespace tndm {

void PetscLinearSolver::warmup() {
    PC pc;
    CHKERRTHROW(KSPSetUp(ksp_));
    CHKERRTHROW(KSPSetUpOnBlocks(ksp_));
    CHKERRTHROW(KSPGetPC(ksp_, &pc));
    warmup_sub_pcs(pc);
}

void PetscLinearSolver::warmup_sub_pcs(PC pc) {
    PCType type;
    PCGetType(pc, &type);
    switch (fnv1a(type)) {
    case HASH_DEF(PCCOMPOSITE):
        warmup_composite(pc);
        break;
    default:
        break;
    };
}

void PetscLinearSolver::warmup_composite(PC pc) {
    PetscInt nc;
    CHKERRTHROW(PCCompositeGetNumberPC(pc, &nc));
    for (PetscInt n = 0; n < nc; ++n) {
        PC sub;
        CHKERRTHROW(PCCompositeGetPC(pc, n, &sub));
        CHKERRTHROW(PCSetUp(sub));
        CHKERRTHROW(PCSetUpOnBlocks(sub));
        warmup_sub_pcs(sub);
    }
}

void PetscLinearSolver::dump() const {
    PetscViewer viewer;
    CHKERRTHROW(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
    CHKERRTHROW(PetscViewerSetType(viewer, PETSCVIEWERBINARY));
    CHKERRTHROW(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));

    CHKERRTHROW(PetscViewerFileSetName(viewer, "A.bin"));
    CHKERRTHROW(MatView(A_->mat(), viewer));

    CHKERRTHROW(PetscViewerFileSetName(viewer, "b.bin"));
    CHKERRTHROW(VecView(b_->vec(), viewer));

    CHKERRTHROW(PetscViewerDestroy(&viewer));
}

} // namespace tndm
