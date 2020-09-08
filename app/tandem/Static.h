#ifndef STATIC_20200825_H
#define STATIC_20200825_H

#include <petscsystypes.h>

namespace tndm {

struct Config;
PetscErrorCode solveStaticProblem(Config const& cfg);

} // namespace tndm

#endif // STATIC_20200825_H
