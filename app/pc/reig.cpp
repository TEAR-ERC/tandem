#include "reig.h"
#include <petscpc.h>
#include <petscpctypes.h>

extern "C" {
#include "reig_impl.h"
}
#include <petsc/private/pcimpl.h>

PetscErrorCode PCApply_reig(PC pc, Vec x, Vec y) {
    PetscFunctionBegin;
    PC comp = reinterpret_cast<PC_reig*>(pc->data)->comp;
    PCApply(comp, x, y);
    PetscFunctionReturn(0);
}

PetscErrorCode PCApplyTranspose_reig(PC pc, Vec x, Vec y) {
    PetscFunctionBegin;
    PC comp = reinterpret_cast<PC_reig*>(pc->data)->comp;
    PCApplyTranspose(comp, x, y);
    PetscFunctionReturn(0);
}

PetscErrorCode PCSetUp_reig(PC pc) {
    PetscFunctionBegin;
    PC comp = reinterpret_cast<PC_reig*>(pc->data)->comp;
    CHKERRQ(PCSetOperators(comp, pc->mat, pc->pmat));
    CHKERRQ(PCSetUp(comp));

    PC pcA, pcD;
    CHKERRQ(PCCompositeGetPC(comp, 0, &pcA));
    CHKERRQ(PCCompositeGetPC(comp, 1, &pcD));
    CHKERRQ(PCSetUp(pcA));
    CHKERRQ(PCSetUp(pcD));
    CHKERRQ(PCConfigure_MGDef(pcD, pc->pmat, 3, 1));

    PetscFunctionReturn(0);
}

PetscErrorCode PCReset_reig(PC pc) {
    PetscFunctionBegin;
    PC comp = reinterpret_cast<PC_reig*>(pc->data)->comp;
    PCReset(comp);
    PetscFunctionReturn(0);
}

PetscErrorCode PCDestroy_reig(PC pc) {
    PetscFunctionBegin;
    PC comp = reinterpret_cast<PC_reig*>(pc->data)->comp;
    PCDestroy(&comp);
    CHKERRQ(PetscFree(pc->data));
    PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_reig(PetscOptionItems* options, PC pc) {
    PetscFunctionBegin;
    PC comp = reinterpret_cast<PC_reig*>(pc->data)->comp;
    PCCompositeSetType(comp, PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE);

    PC pcA, pcD;
    CHKERRQ(PCCompositeAddPC(comp, PCASM));
    CHKERRQ(PCCompositeAddPC(comp, PCSHELL));
    CHKERRQ(PCCompositeGetPC(comp, 0, &pcA));
    CHKERRQ(PCCompositeGetPC(comp, 1, &pcD));
    CHKERRQ(PCSetFromOptions(pcA));
    CHKERRQ(PCSetFromOptions(pcD));

    PetscFunctionReturn(0);
}

PetscErrorCode PCView_reig(PC pc, PetscViewer viewer) {
    PC comp = reinterpret_cast<PC_reig*>(pc->data)->comp;
    PCView(comp, viewer);
    PetscFunctionReturn(0);
}

PetscErrorCode PCCreate_reig(PC pc) {
    PC_reig* reig;

    PetscFunctionBegin;
    CHKERRQ(PetscNewLog(pc, &reig));

    pc->ops->apply = PCApply_reig;
    pc->ops->applytranspose = PCApplyTranspose_reig;
    pc->ops->setup = PCSetUp_reig;
    pc->ops->reset = PCReset_reig;
    pc->ops->destroy = PCDestroy_reig;
    pc->ops->setfromoptions = PCSetFromOptions_reig;
    pc->ops->view = PCView_reig;
    pc->ops->applyrichardson = 0;

    pc->data = (void*)reig;

    CHKERRQ(PCCreate(PetscObjectComm((PetscObject)pc), &reig->comp));
    CHKERRQ(PCSetType(reig->comp, PCCOMPOSITE));

    PetscFunctionReturn(0);
}
