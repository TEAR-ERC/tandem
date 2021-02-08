#ifndef REIG_IMPL_20210208_H
#define REIG_IMPL_20210208_H

#include <petscmat.h>
#include <petscpc.h>
#include <petscsys.h>

PetscErrorCode PCConfigure_MGDef(PC pc, Mat A, PetscInt nev, PetscInt type);

#endif // REIG_IMPL_20210208_H
