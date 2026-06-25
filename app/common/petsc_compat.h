#ifndef TANDEM_PETSC_COMPAT_H
#define TANDEM_PETSC_COMPAT_H

#include <petsc/private/kspimpl.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscversion.h>

/**
 * 1. Abstract Option Passing Types & Layouts
 * PETSc 3.23 changed PetscOptionItems from a pointer to a value pass.
 */
#if PETSC_VERSION_LT(3, 23, 0)
typedef PetscOptionItems* TandemPetscOptions;
#define TandemPetscOptionsBegin(obj, mess, pre) PetscOptionsHeadBegin(PetscOptionsObject, mess)
#define TandemPetscOptionsEnd() PetscOptionsHeadEnd()
#else
typedef PetscOptionItems TandemPetscOptions;
#define TandemPetscOptionsBegin(obj, mess, pre)                                                    \
    PetscOptionsBegin(PetscObjectComm((PetscObject)obj), ((PetscObject)obj)->prefix, mess, pre)
#define TandemPetscOptionsEnd() PetscOptionsEnd()
#endif

/**
 * 2. Abstract Container Destruction API
 */
static inline PetscErrorCode TandemPetscContainerSetDestroy(PetscContainer container) {
#if PETSC_VERSION_LT(3, 23, 0)
    return PetscContainerSetUserDestroy(container, PetscContainerUserDestroyDefault);
#else
    return PetscContainerSetCtxDestroy(container, PetscContainerCtxDestroyDefault);
#endif
}

/**
 * 3. Abstract Private/Internal KSP Noisy Setup
 */
static inline PetscErrorCode TandemKSPSetNoisy(KSP kspest, Vec work_bg, Vec work_noisy) {
#if PETSC_VERSION_LT(3, 23, 0)
    (void)kspest;
    (void)work_bg;
    return KSPSetNoisy_Private(work_noisy);
#else
    Mat A;
    PetscErrorCode ierr = KSPGetOperators(kspest, &A, NULL);
    if (ierr)
        return ierr;
    return KSPSetNoisy_Private(A, work_noisy);
#endif
}

#endif // TANDEM_PETSC_COMPAT_H
