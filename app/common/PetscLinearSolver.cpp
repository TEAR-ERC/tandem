#include "PetscLinearSolver.h"
#include "common/PetscUtil.h"
#include <petscpc.h>

namespace tndm {

void PetscLinearSolver::warmup() { warmup_ksp(ksp_); }

void PetscLinearSolver::warmup_ksp(KSP ksp) {
    PC pc;
    CHKERRTHROW(KSPSetUp(ksp));
    CHKERRTHROW(KSPSetUpOnBlocks(ksp));
    CHKERRTHROW(KSPGetPC(ksp, &pc));
    warmup_sub_pcs(pc);
}

void PetscLinearSolver::warmup_sub_pcs(PC pc) {
    PCType type;
    PCGetType(pc, &type);
    switch (fnv1a(type)) {
    case HASH_DEF(PCCOMPOSITE):
        warmup_composite(pc);
        break;
    case HASH_DEF(PCMG):
        warmup_mg(pc);
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

void PetscLinearSolver::warmup_mg(PC pc) {
    PetscInt levels;
    CHKERRTHROW(PCMGGetLevels(pc, &levels));
    for (PetscInt level = 0; level < levels; ++level) {
        KSP smoother;
        CHKERRTHROW(PCMGGetSmoother(pc, level, &smoother));
        warmup_ksp(smoother);
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
