#ifndef TS_UTIL_20211202_H
#define TS_UTIL_20211202_H

#include <petscts.h>


PetscErrorCode ts_checkpoint_configure(TS ts);
PetscErrorCode ts_checkpoint(TS ts);
PetscErrorCode ts_checkpoint_restart(TS ts);

#endif // TS_UTIL_20211202_H
