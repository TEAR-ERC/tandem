#include "eigdeflate.h"
#include "reig_aux.h"

#include <petsc/private/pcimpl.h>
#include <petscksp.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsystypes.h>

typedef struct {
    KSP reig, smooth;
    PetscReal e_min, e_max, factor, alpha;
    PetscInt nev, nev_oversample, power_its, npre, npost;
    Vec eigs;
    Mat Q;

    Vec r, rc, rc_red;
    VecScatter scatter;
} PC_eigdeflate;

PetscErrorCode PCApply_eigdeflate(PC pc, Vec x, Vec y) {
    PC_eigdeflate* ctx;
    Mat A;

    ctx = (PC_eigdeflate*)pc->data;
    A = pc->pmat;

    CHKERRQ(VecZeroEntries(y));

    if (ctx->npre > 0) {
        CHKERRQ(
            KSPSetTolerances(ctx->smooth, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, ctx->npre));
        CHKERRQ(KSPSetInitialGuessNonzero(ctx->smooth, PETSC_TRUE));
        CHKERRQ(KSPSolve(ctx->smooth, x, y));
        CHKERRQ(MatMult(A, y, ctx->r));
        CHKERRQ(VecAYPX(ctx->r, -1.0, x));
        CHKERRQ(MatMultTranspose(ctx->Q, ctx->r, ctx->rc));
    } else {
        CHKERRQ(MatMultTranspose(ctx->Q, x, ctx->rc));
    }

    CHKERRQ(VecScatterBegin(ctx->scatter, ctx->rc, ctx->rc_red, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(ctx->scatter, ctx->rc, ctx->rc_red, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecPointwiseDivide(ctx->rc_red, ctx->rc_red, ctx->eigs));
    CHKERRQ(MatMultRedundant_MatDenseVecSeq(ctx->Q, ctx->rc_red, y));

    if (ctx->npost > 0) {
        CHKERRQ(
            KSPSetTolerances(ctx->smooth, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, ctx->npost));
        CHKERRQ(KSPSetInitialGuessNonzero(ctx->smooth, PETSC_TRUE));
        CHKERRQ(KSPSolve(ctx->smooth, x, y));
    }

    PetscFunctionReturn(0);
}

PetscErrorCode PCSetUp_eigdeflate(PC pc) {
    PC_eigdeflate* ctx;
    Mat A;
    const char* prefix;

    ctx = (PC_eigdeflate*)pc->data;
    A = pc->pmat;

    if (!ctx->reig) {
        CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)A), &ctx->reig));
        CHKERRQ(PCGetOptionsPrefix(pc, &prefix));
        CHKERRQ(KSPSetOptionsPrefix(ctx->reig, prefix));
        CHKERRQ(KSPAppendOptionsPrefix(ctx->reig, "eigdeflate_"));
    }
    CHKERRQ(KSPSetOperators(ctx->reig, A, A));
    if (pc->setfromoptionscalled && !pc->setupcalled) {
        CHKERRQ(KSPSetFromOptions(ctx->reig));
    }

    CHKERRQ(RandEigsMin(ctx->reig, ctx->nev, ctx->nev_oversample, ctx->power_its, NULL, &ctx->eigs,
                        &ctx->Q));
    {
        PetscReal* _e;
        CHKERRQ(VecGetArray(ctx->eigs, &_e));
        ctx->e_min = _e[0];
        CHKERRQ(VecRestoreArray(ctx->eigs, &_e));
    }
    {
        Vec eigs_max = NULL;
        Mat Q_max = NULL;
        PetscReal* _e;
        PetscInt len;

        CHKERRQ(RandEigsMax(A, 1, 5, ctx->power_its, NULL, &eigs_max, &Q_max));
        CHKERRQ(VecGetSize(eigs_max, &len));
        CHKERRQ(VecGetArray(eigs_max, &_e));
        ctx->e_max = _e[len - 1];
        CHKERRQ(VecRestoreArray(eigs_max, &_e));
        CHKERRQ(VecDestroy(&eigs_max));
        CHKERRQ(MatDestroy(&Q_max));
    }

    ctx->alpha = 2.0 * ctx->factor / (ctx->e_min + ctx->e_max);

    if (!ctx->rc && !ctx->r) {
        CHKERRQ(MatCreateVecs(ctx->Q, &ctx->rc, &ctx->r));
    }
    if (!ctx->scatter && !ctx->rc_red) {
        CHKERRQ(VecScatterCreateToAll(ctx->rc, &ctx->scatter, &ctx->rc_red));
    }

    if (ctx->npre > 0 || ctx->npost > 0) {
        if (!ctx->smooth) {
            CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)A), &ctx->smooth));
            CHKERRQ(PCGetOptionsPrefix(pc, &prefix));
            CHKERRQ(KSPSetOptionsPrefix(ctx->smooth, prefix));
            CHKERRQ(KSPAppendOptionsPrefix(ctx->smooth, "eigdeflate_smooth_"));
        }
        CHKERRQ(KSPSetOperators(ctx->smooth, A, A));
        if (pc->setfromoptionscalled && !pc->setupcalled) {
            CHKERRQ(KSPSetFromOptions(ctx->smooth));
        }
        CHKERRQ(KSPRichardsonSetScale(ctx->smooth, ctx->alpha));
    }

    PetscFunctionReturn(0);
}

PetscErrorCode PCReset_eigdeflate(PC pc) {
    PC_eigdeflate* ctx = (PC_eigdeflate*)pc->data;

    CHKERRQ(KSPReset(ctx->reig));
    CHKERRQ(MatDestroy(&ctx->Q));
    CHKERRQ(VecDestroy(&ctx->r));

    PetscFunctionReturn(0);
}

PetscErrorCode PCDestroy_eigdeflate(PC pc) {
    PC_eigdeflate* ctx = (PC_eigdeflate*)pc->data;

    PCReset_eigdeflate(pc);

    CHKERRQ(KSPDestroy(&ctx->reig));
    if (ctx->npre > 0 || ctx->npost > 0) {
        CHKERRQ(KSPDestroy(&ctx->smooth));
    }
    CHKERRQ(VecDestroy(&ctx->eigs));
    CHKERRQ(VecDestroy(&ctx->rc));
    CHKERRQ(VecDestroy(&ctx->rc_red));
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
    PetscOptionsBoundedInt("-pc_eigdeflate_npre", "Number of pre-smoothing iterations", "",
                           ctx->npre, &ctx->npre, &flg, 0);
    PetscOptionsBoundedInt("-pc_eigdeflate_npost", "Number of post-smoothing iterations", "",
                           ctx->npost, &ctx->npost, &flg, 0);
    PetscOptionsReal("-pc_eigdeflate_relax_factor", "Relaxation factor", "", ctx->factor,
                     &ctx->factor, &flg);
    PetscOptionsTail();
    PetscFunctionReturn(0);
}

PetscErrorCode PCView_eigdeflate(PC pc, PetscViewer viewer) {
    PC_eigdeflate* ctx = (PC_eigdeflate*)pc->data;

    PetscViewerASCIIPrintf(viewer, "num. eigenvectors: %D\n", ctx->nev);
    PetscViewerASCIIPrintf(viewer, "emin: %+1.4e\n", ctx->e_min);
    PetscViewerASCIIPrintf(viewer, "emax: %+1.4e\n", ctx->e_max);
    PetscViewerASCIIPrintf(viewer, "Randomized eigenvalue calculation\n", ctx->nev_oversample);
    PetscViewerASCIIPrintf(viewer, "over sampling: %D\n", ctx->nev_oversample);
    PetscViewerASCIIPrintf(viewer, "power iterations: %D\n", ctx->power_its);
    PetscViewerASCIIPrintf(viewer, "pre-smooth iterations:  %D\n", ctx->npre);
    PetscViewerASCIIPrintf(viewer, "post-smooth iterations: %D\n", ctx->npost);
    if (ctx->npre > 0 || ctx->npost > 0) {
        PetscViewerASCIIPrintf(viewer, "optimal relaxation: %+1.4e\n", ctx->alpha);
    }
    PetscViewerASCIIPushTab(viewer);
    CHKERRQ(KSPView(ctx->reig, viewer));
    if (ctx->npre > 0 || ctx->npost > 0) {
        CHKERRQ(KSPView(ctx->smooth, viewer));
    }
    PetscViewerASCIIPopTab(viewer);

    PetscFunctionReturn(0);
}

PetscErrorCode PCCreate_eigdeflate(PC pc) {
    PC_eigdeflate* edef;

    CHKERRQ(PetscNewLog(pc, &edef));
    pc->data = (void*)edef;

    edef->reig = NULL;
    edef->smooth = NULL;
    edef->e_min = 0.0;
    edef->e_max = 0.0;
    edef->factor = 1.0;
    edef->alpha = 0.0;
    edef->nev = 2;
    edef->nev_oversample = 2;
    edef->power_its = 1;
    edef->npre = 0;
    edef->npost = 0;
    edef->eigs = NULL;
    edef->Q = NULL;
    edef->r = NULL;
    edef->rc = NULL;
    edef->rc_red = NULL;
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
