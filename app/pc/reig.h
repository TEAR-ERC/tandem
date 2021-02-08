#ifndef REIG_20210208_H
#define REIG_20210208_H

#include <petscpc.h>
#include <petscsys.h>

struct PC_reig {
    PC comp;
};

PetscErrorCode PCCreate_reig(PC pc);

#endif // REIG_20210208_H
