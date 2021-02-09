#include "eigdeflate.h"
#include "reig_aux.h"

#include <petsc/private/pcimpl.h>
#include <petscpc.h>
#include <petscpctypes.h>

typedef struct {
    KSP reig;
    PetscReal e_min;
    PetscInt nev, nev_oversample, power_its;
    Vec eigs;
    Mat Q;

    Vec r, e, rc, rc_red, ec_red;
    VecScatter scatter;
} PC_eigdeflate;

PetscErrorCode PCApply_eigdeflate(PC pc, Vec x, Vec y) {
    PC_eigdeflate* ctx = (PC_eigdeflate*)pc->data;

    CHKERRQ(VecZeroEntries(y));
    CHKERRQ(MatMultTranspose(ctx->Q, x, ctx->rc));
    CHKERRQ(VecScatterBegin(ctx->scatter, ctx->rc, ctx->rc_red, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(ctx->scatter, ctx->rc, ctx->rc_red, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecCopy(ctx->rc_red, ctx->ec_red));
    CHKERRQ(VecPointwiseDivide(ctx->ec_red, ctx->ec_red, ctx->eigs));
    CHKERRQ(MatMultRedundant_MatDenseVecSeq(ctx->Q, ctx->ec_red, ctx->e));
    CHKERRQ(VecAXPY(y, 1.0, ctx->e));

    PetscFunctionReturn(0);
}

PetscErrorCode PCSetUp_eigdeflate(PC pc) {
    PC_eigdeflate* ctx;
    PetscInt M;
    Mat A;
    const char* prefix;

    ctx = (PC_eigdeflate*)pc->data;
    A = pc->pmat;

    CHKERRQ(MatGetSize(A, &M, NULL));

    if (!ctx->reig) {
        CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)A), &ctx->reig));
        PCGetOptionsPrefix(pc, &prefix);
        CHKERRQ(KSPSetOptionsPrefix(ctx->reig, prefix));
        CHKERRQ(KSPAppendOptionsPrefix(ctx->reig, "eigdeflate_"));
        CHKERRQ(KSPSetOperators(ctx->reig, A, A));
        CHKERRQ(KSPSetFromOptions(ctx->reig));
    }

    CHKERRQ(RandEigsMin_MPI_2(ctx->reig, ctx->nev, ctx->nev_oversample, ctx->power_its, NULL,
                              &ctx->eigs, &ctx->Q));

    {
        PetscReal* _e;
        CHKERRQ(VecGetArray(ctx->eigs, &_e));
        ctx->e_min = _e[0];
        CHKERRQ(VecRestoreArray(ctx->eigs, &_e));
    }

    if (!ctx->rc && !ctx->r) {
        CHKERRQ(MatCreateVecs(ctx->Q, &ctx->rc, &ctx->r));
    }
    if (!ctx->e) {
        CHKERRQ(MatCreateVecs(ctx->Q, NULL, &ctx->e));
    }
    if (!ctx->scatter && !ctx->rc_red) {
        CHKERRQ(VecScatterCreateToAll(ctx->rc, &ctx->scatter, &ctx->rc_red));
    }
    if (!ctx->ec_red) {
        CHKERRQ(VecDuplicate(ctx->eigs, &ctx->ec_red));
    }

    PetscFunctionReturn(0);
}

PetscErrorCode PCReset_eigdeflate(PC pc) {
    PC_eigdeflate* ctx = (PC_eigdeflate*)pc->data;

    CHKERRQ(KSPReset(ctx->reig));
    CHKERRQ(MatDestroy(&ctx->Q));
    CHKERRQ(VecDestroy(&ctx->r));
    CHKERRQ(VecDestroy(&ctx->e));

    PetscFunctionReturn(0);
}

PetscErrorCode PCDestroy_eigdeflate(PC pc) {
    PC_eigdeflate* ctx = (PC_eigdeflate*)pc->data;

    PCReset_eigdeflate(pc);

    CHKERRQ(KSPDestroy(&ctx->reig));
    CHKERRQ(VecDestroy(&ctx->eigs));
    CHKERRQ(VecDestroy(&ctx->rc));
    CHKERRQ(VecDestroy(&ctx->rc_red));
    CHKERRQ(VecDestroy(&ctx->ec_red));
    CHKERRQ(VecScatterDestroy(&ctx->scatter));

    CHKERRQ(PetscFree(ctx));

    PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_eigdeflate(PetscOptionItems* PetscOptionsObject, PC pc) {
    PC_eigdeflate* ctx;
    PetscInt M;
    PetscBool flg;

    ctx = (PC_eigdeflate*)pc->data;

    PetscOptionsHead(PetscOptionsObject, "Eigdeflate options");
    PetscOptionsBoundedInt("-pc_eigdeflate_nev", "Number of deflated eigenvalues", "", ctx->nev,
                           &ctx->nev, &flg, 1);
    PetscOptionsBoundedInt("-pc_eigdeflate_nev_oversample",
                           "Number of oversampled evs to enhance accuracy", "", ctx->nev_oversample,
                           &ctx->nev_oversample, &flg, 0);
    PetscOptionsBoundedInt("-pc_eigdeflate_power_its", "Number of power iterations", "",
                           ctx->power_its, &ctx->power_its, &flg, 0);
    PetscOptionsTail();
    PetscFunctionReturn(0);
}

PetscErrorCode PCView_eigdeflate(PC pc, PetscViewer viewer) {
    PC_eigdeflate* ctx = (PC_eigdeflate*)pc->data;

    PetscViewerASCIIPushTab(viewer);
    PetscViewerASCIIPrintf(viewer, "num. eigenvectors: %D\n", ctx->nev);
    PetscViewerASCIIPrintf(viewer, "emin: %+1.4e\n", ctx->e_min);
    PetscViewerASCIIPrintf(viewer, "Randomized eigenvalue calculation\n", ctx->nev_oversample);
    PetscViewerASCIIPrintf(viewer, "over sampling: %D\n", ctx->nev_oversample);
    PetscViewerASCIIPrintf(viewer, "power iterations: %D\n", ctx->power_its);
    CHKERRQ(KSPView(ctx->reig, viewer));
    PetscViewerASCIIPopTab(viewer);

    PetscFunctionReturn(0);
}

PetscErrorCode PCCreate_eigdeflate(PC pc) {
    PC_eigdeflate* edef;

    CHKERRQ(PetscNewLog(pc, &edef));
    pc->data = (void*)edef;

    edef->reig = NULL;
    edef->e_min = 0.0;
    edef->nev = 2;
    edef->nev_oversample = 2;
    edef->power_its = 1;
    edef->eigs = NULL;
    edef->Q = NULL;
    edef->r = NULL;
    edef->e = NULL;
    edef->rc = NULL;
    edef->rc_red = NULL;
    edef->ec_red = NULL;
    edef->scatter = NULL;

    pc->ops->apply = PCApply_eigdeflate;
    pc->ops->applytranspose = PCApply_eigdeflate;
    pc->ops->setup = PCSetUp_eigdeflate;
    pc->ops->reset = PCReset_eigdeflate;
    pc->ops->destroy = PCDestroy_eigdeflate;
    pc->ops->setfromoptions = PCSetFromOptions_eigdeflate;
    pc->ops->view = PCView_eigdeflate;
    pc->ops->applyrichardson = NULL;
    pc->ops->applysymmetricleft = NULL;
    pc->ops->applysymmetricleft = NULL;

    PetscFunctionReturn(0);
}
