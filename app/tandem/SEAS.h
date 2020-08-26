#ifndef SEAS_20200825_H
#define SEAS_20200825_H

#include "Config.h"

#include <petscsystypes.h>

namespace tndm {

PetscErrorCode solveSEASProblem(Config const& cfg);

}

#endif // SEAS_20200825_H
