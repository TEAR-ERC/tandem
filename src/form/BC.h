#ifndef BC_20200910_H
#define BC_20200910_H
#include <iostream>

namespace tndm {

enum class BC : int { None = 0, Natural = 1, Fault = 3, Dirichlet = 5, Unknown = -1 };

// TODO: Choose these ranges as decided
const long DIRICHLET_MIN = 5;
const long DIRICHLET_MAX = 5;
const long DIRICHLET_DEFAULT = 5;
const long FAULT_MIN = 6;
const long FAULT_MAX = 1000;
const long FAULT_DEFAULT = 3;
const long NATURAL_MIN = 1;
const long NATURAL_MAX = 1;
const long NATURAL_DEFAULT = 1;

inline BC classifyBC(long int tag) {
    if (tag == 0)
        return BC::None;
    if (tag == DIRICHLET_DEFAULT || (tag >= DIRICHLET_MIN && tag <= DIRICHLET_MAX))
        return BC::Dirichlet;
    if (tag == FAULT_DEFAULT || (tag >= FAULT_MIN && tag <= FAULT_MAX))
        return BC::Fault;
    if (tag == NATURAL_DEFAULT || (tag >= NATURAL_MIN && tag <= NATURAL_MAX))
        return BC::Natural;
    return BC::Unknown;
}
} // namespace tndm
#endif // BC_20200910_H
