#include <petsc/private/kspimpl.h>
#include <petsc/private/pcimpl.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscpc.h>

typedef struct {
    PetscReal alpha, beta; /* Coefficients of w(x) = (1-x)^\alpha (1+x)^\beta */
    PetscReal emax;        /* maximum eigenvalue */
    KSP kspest;            /* KSP used to estimate eigenvalues */
    PetscInt eststeps;     /* number of kspest steps in KSP used to estimate eigenvalues */
    PetscReal estbias;     /* maximum eigenvalue is scaled with this factor */
} KSP_LSPoly;

typedef struct {
    PetscReal alpha, beta;
    PetscReal emax;       /* Maximum eigenvalue */
    PetscReal P0n_1, P0n; /* P_{n-1|n}^{\alpha,\beta+1}(0) */
    PetscInt n;
} ResidualP;

void ResidualP_Init(ResidualP* rp, PetscReal alpha, PetscReal beta, PetscReal emax) {
    rp->alpha = alpha;
    rp->beta = beta + 1.0;
    rp->emax = emax;
    rp->P0n_1 = 1.0;
    rp->P0n = 1.0;
    rp->n = -1;
}

/* Coefficients for the recurrence relation of the residual polynomial
 *
 * R_{n+1}(\lambda) = (\psi_n + \phi_n \lambda) R_n(\lambda) - \xi_n R_{n-1}(\lambda)
 *
 * */
void ResidualP_Next(ResidualP* rp, PetscReal* psi, PetscReal* phi, PetscReal* xi) {
    PetscReal a, b, n2ab, tmp;
    PetscInt n;
    PetscReal phi0, psi0, xi0, P0n1;

    a = rp->alpha;
    b = rp->beta;
    n = ++rp->n;

    /* Compute recurrence coefficients of Jacobi polynomial on [-1,1]*/
    n2ab = 2 * n + a + b;
    tmp = 2 * (n + 1) * (n + a + b + 1);
    psi0 = 0.0;
    if (fabs(a) != fabs(b)) {
        psi0 = (n2ab + 1) * (a * a - b * b) / (tmp * n2ab);
    }
    phi0 = (n2ab + 1) * (n2ab + 2) / tmp;
    xi0 = 0.0;
    if (n > 0) {
        xi0 = 2 * (n + a) * (n + b) * (n2ab + 2) / (tmp * n2ab);
    }

    /* Shift interval on [0, emax] */
    psi0 = psi0 - phi0;
    phi0 = 2.0 * phi0 / rp->emax;

    /* Compute normalized coefficients */
    tmp = (psi0 - xi0 * rp->P0n_1 / rp->P0n);
    *psi = psi0 / tmp;
    *phi = phi0 / tmp;
    tmp = (psi0 * rp->P0n / rp->P0n_1 - xi0);
    *xi = xi0 / tmp;

    /* Evolve P_n(0) */
    P0n1 = psi0 * rp->P0n - xi0 * rp->P0n_1;
    rp->P0n_1 = rp->P0n;
    rp->P0n = P0n1;
}

static PetscErrorCode KSPReset_LSPoly(KSP ksp) {
    KSP_LSPoly* cheb = (KSP_LSPoly*)ksp->data;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (cheb->kspest) {
        ierr = KSPReset(cheb->kspest);
        CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

/*
 * Must be passed a KSP solver that has "converged", with KSPSetComputeEigenvalues() called before
 * the solve
 */
static PetscErrorCode KSPLSPolyComputeExtremeEigenvalues_Private(KSP kspest, PetscReal* emax) {
    PetscInt n, neig;
    PetscReal *re, *im, max;

    PetscFunctionBegin;
    CHKERRQ(KSPGetIterationNumber(kspest, &n));
    CHKERRQ(PetscMalloc2(n, &re, n, &im));
    CHKERRQ(KSPComputeEigenvalues(kspest, n, re, im, &neig));
    max = PETSC_MIN_REAL;
    for (n = 0; n < neig; n++) {
        max = PetscMax(max, re[n]);
    }
    CHKERRQ(PetscFree2(re, im));
    *emax = max;
    PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_LSPoly(KSP ksp) {
    KSP_LSPoly* cheb = (KSP_LSPoly*)ksp->data;
    PetscFunctionBegin;
    CHKERRQ(KSPSetWorkVecs(ksp, 3));
    if (!cheb->kspest) {
        CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)ksp), &cheb->kspest));
        CHKERRQ(KSPSetErrorIfNotConverged(cheb->kspest, ksp->errorifnotconverged));
        CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)cheb->kspest, (PetscObject)ksp, 1));
        /* use PetscObjectSet/AppendOptionsPrefix() instead of KSPSet/AppendOptionsPrefix() so
         * that the PC prefix is not changed */
        CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)cheb->kspest, ((PetscObject)ksp)->prefix));
        CHKERRQ(PetscObjectAppendOptionsPrefix((PetscObject)cheb->kspest, "esteig_"));
        CHKERRQ(KSPSetSkipPCSetFromOptions(cheb->kspest, PETSC_TRUE));
        CHKERRQ(KSPSetComputeEigenvalues(cheb->kspest, PETSC_TRUE));
        CHKERRQ(KSPSetType(cheb->kspest, KSPCG));
        CHKERRQ(
            KSPSetTolerances(cheb->kspest, 1.e-12, PETSC_DEFAULT, PETSC_DEFAULT, cheb->eststeps));
    }
    if (cheb->kspest) {
        PetscReal max = 0.0;
        KSPConvergedReason reason;
        CHKERRQ(KSPSetPC(cheb->kspest, ksp->pc));

        CHKERRQ(KSPSetNoisy_Private(ksp->work[1]));
        CHKERRQ(KSPSolve(cheb->kspest, ksp->work[1], ksp->work[0]));
        CHKERRQ(KSPGetConvergedReason(cheb->kspest, &reason));
        if (reason == KSP_DIVERGED_ITS) {
            CHKERRQ(PetscInfo(ksp, "Eigen estimator ran for prescribed number of iterations\n"));
        } else if (reason == KSP_DIVERGED_PC_FAILED) {
            PetscInt its;
            PCFailedReason pcreason;

            CHKERRQ(KSPGetIterationNumber(cheb->kspest, &its));
            if (ksp->normtype == KSP_NORM_NONE) {
                PetscInt sendbuf, recvbuf;
		CHKERRQ(PCReduceFailedReason(ksp->pc));
            }
            CHKERRQ(PCGetFailedReason(ksp->pc, &pcreason));
            ksp->reason = KSP_DIVERGED_PC_FAILED;
            CHKERRQ(PetscInfo(ksp, "Eigen estimator failed: %s %s at iteration %" PetscInt_FMT "",
                               KSPConvergedReasons[reason], PCFailedReasons[pcreason], its));
            PetscFunctionReturn(0);
        } else if (reason == KSP_CONVERGED_RTOL || reason == KSP_CONVERGED_ATOL) {
            CHKERRQ(PetscInfo(ksp, "Eigen estimator converged prematurely. Should not happen "
                                   "except for small or low rank problem\n"));
        } else if (reason < 0) {
            CHKERRQ(PetscInfo(ksp, "Eigen estimator failed %s, using estimates anyway\n",
                               KSPConvergedReasons[reason]));
        }

        CHKERRQ(KSPLSPolyComputeExtremeEigenvalues_Private(cheb->kspest, &max));
        CHKERRQ(KSPSetPC(cheb->kspest, NULL));

        cheb->emax = max;
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_LSPoly(KSP ksp, PetscOptionItems* PetscOptionsObject) {
    KSP_LSPoly* cheb = (KSP_LSPoly*)ksp->data;

    PetscFunctionBegin;
    PetscOptionsHeadBegin(PetscOptionsObject, "KSP LSPoly Options");
    CHKERRQ(PetscOptionsInt("-ksp_lspoly_esteig_steps", "Number of est steps in LSPoly", "",
                            cheb->eststeps, &cheb->eststeps, NULL));
    CHKERRQ(PetscOptionsReal("-ksp_lspoly_esteig_bias",
                             "Maximum eigenvalue is multiplied with this factor", "", cheb->estbias,
                             &cheb->estbias, NULL));
    CHKERRQ(PetscOptionsReal("-ksp_lspoly_alpha", "Weighting function (1-x)^alpha", "", cheb->alpha,
                             &cheb->alpha, NULL));
    CHKERRQ(PetscOptionsReal("-ksp_lspoly_beta", "Weighting function (1+x)^beta", "", cheb->beta,
                             &cheb->beta, NULL));
    PetscOptionsHeadEnd();
    PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_LSPoly(KSP ksp) {
    KSP_LSPoly* lspoly = (KSP_LSPoly*)ksp->data;
    PetscInt i;
    PetscReal rnorm = 0.0, psi, phi, xi;
    Vec b, delta, r, r_pre, x;
    Mat Amat, Pmat;
    PetscBool diagonalscale;
    ResidualP rp;

    PetscFunctionBegin;
    CHKERRQ(PCGetDiagonalScale(ksp->pc, &diagonalscale));
    if (diagonalscale)
        SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP,
                 "Krylov method %s does not support diagonal scaling",
                 ((PetscObject)ksp)->type_name);

    CHKERRQ(PCGetOperators(ksp->pc, &Amat, &Pmat));
    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its = 0;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    x = ksp->vec_sol;
    b = ksp->vec_rhs;
    r = ksp->work[0];
    r_pre = ksp->work[1];
    delta = ksp->work[2];
    VecSet(delta, 0.0);

    ResidualP_Init(&rp, lspoly->alpha, lspoly->beta, lspoly->estbias * lspoly->emax);

    for (i = 0; i < ksp->max_it; i++) {
        CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
        ++ksp->its;
        CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));

        /*  r = b - Ax   */
        if (x == 0 && ksp->guess_zero) {
            CHKERRQ(VecCopy(b, r));
        } else {
            CHKERRQ(KSP_MatMult(ksp, Amat, x, r));
            CHKERRQ(VecAYPX(r, -1.0, b));
        }

        /*  r_pre = B^{-1}r  */
        CHKERRQ(KSP_PCApply(ksp, r, r_pre));

        /* calculate residual norm if requested */
        if (ksp->normtype) {
            switch (ksp->normtype) {
            case KSP_NORM_PRECONDITIONED:
                CHKERRQ(VecNorm(r_pre, NORM_2, &rnorm));
                break;
            case KSP_NORM_UNPRECONDITIONED:
            case KSP_NORM_NATURAL:
                CHKERRQ(VecNorm(r, NORM_2, &rnorm));
                break;
            default:
                rnorm = 0.0;
                break;
            }
            KSPCheckNorm(ksp, rnorm);
            CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
            ksp->rnorm = rnorm;
            CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
            CHKERRQ(KSPLogResidualHistory(ksp, rnorm));
            CHKERRQ(KSPMonitor(ksp, i, rnorm));
            CHKERRQ((*ksp->converged)(ksp, i, rnorm, &ksp->reason, ksp->cnvP));
            if (ksp->reason)
                break;
        }
        CHKERRQ(KSPLogErrorHistory(ksp));

        ResidualP_Next(&rp, &psi, &phi, &xi);

        CHKERRQ(VecAXPBY(delta, -phi, xi, r_pre));
        CHKERRQ(VecAXPY(x, 1.0, delta));
    }

    if (!ksp->reason) {
        if (ksp->normtype) {
            CHKERRQ(KSP_MatMult(ksp, Amat, x, r)); /*  r = b - Ax    */
            CHKERRQ(VecAYPX(r, -1.0, b));
            switch (ksp->normtype) {
            case KSP_NORM_PRECONDITIONED:
                /*  r_pre = B^{-1}r  */
                CHKERRQ(KSP_PCApply(ksp, r, r_pre));
                CHKERRQ(VecNorm(r_pre, NORM_2, &rnorm));
                break;
            case KSP_NORM_UNPRECONDITIONED:
            case KSP_NORM_NATURAL:
                CHKERRQ(VecNorm(r, NORM_2, &rnorm));
                break;
            default:
                rnorm = 0.0;
                break;
            }
            KSPCheckNorm(ksp, rnorm);
            CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
            ksp->rnorm = rnorm;
            CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
            CHKERRQ(KSPLogResidualHistory(ksp, rnorm));
            CHKERRQ(KSPMonitor(ksp, i, rnorm));
        }
        if (ksp->its >= ksp->max_it) {
            if (ksp->normtype != KSP_NORM_NONE) {
                CHKERRQ((*ksp->converged)(ksp, i, rnorm, &ksp->reason, ksp->cnvP));
                if (!ksp->reason) {
                    ksp->reason = KSP_DIVERGED_ITS;
                }
            } else {
                ksp->reason = KSP_CONVERGED_ITS;
            }
        }
    }

    if (ksp->reason == KSP_CONVERGED_ITS) {
        CHKERRQ(KSPLogErrorHistory(ksp));
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_LSPoly(KSP ksp, PetscViewer viewer) {
    KSP_LSPoly* cheb = (KSP_LSPoly*)ksp->data;
    PetscBool iascii;

    PetscFunctionBegin;
    CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
    if (iascii) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "  eigenvalue estimates used:  max = %g\n",
                                       (double)cheb->emax));
        if (cheb->kspest) {
            CHKERRQ(PetscViewerASCIIPushTab(viewer));
            CHKERRQ(KSPView(cheb->kspest, viewer));
            CHKERRQ(PetscViewerASCIIPopTab(viewer));
        }
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_LSPoly(KSP ksp) {
    KSP_LSPoly* cheb = (KSP_LSPoly*)ksp->data;

    PetscFunctionBegin;
    CHKERRQ(KSPDestroy(&cheb->kspest));
    CHKERRQ(KSPDestroyDefault(ksp));
    PetscFunctionReturn(0);
}

/*MC
     KSPLSPOLY - The preconditioned LSPoly iterative method

   Options Database Keys:
-ksp_lspoly_esteig_steps - number of estimation steps

   Level: beginner

   Notes:
    The LSPoly method requires both the matrix and preconditioner to
          be symmetric positive (semi) definite.
          Only support for left preconditioning.

          LSPoly is configured as a smoother by default, targetting the "upper" part of the
spectrum.

M*/

PETSC_EXTERN PetscErrorCode KSPCreate_LSPoly(KSP ksp) {
    KSP_LSPoly* lspolyP;

    PetscFunctionBegin;
    CHKERRQ(PetscNew(&lspolyP));

    ksp->data = (void*)lspolyP;
    CHKERRQ(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 3));
    CHKERRQ(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 2));
    CHKERRQ(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));
    CHKERRQ(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_RIGHT, 1));

    lspolyP->alpha = 0.;
    lspolyP->beta = 0.;
    lspolyP->emax = 0.;
    lspolyP->estbias = 1.1;

    lspolyP->eststeps = 10;
    ksp->setupnewmatrix = PETSC_TRUE;

    ksp->ops->setup = KSPSetUp_LSPoly;
    ksp->ops->solve = KSPSolve_LSPoly;
    ksp->ops->destroy = KSPDestroy_LSPoly;
    ksp->ops->buildsolution = KSPBuildSolutionDefault;
    ksp->ops->buildresidual = KSPBuildResidualDefault;
    ksp->ops->setfromoptions = KSPSetFromOptions_LSPoly;
    ksp->ops->view = KSPView_LSPoly;
    ksp->ops->reset = KSPReset_LSPoly;

    PetscFunctionReturn(0);
}
