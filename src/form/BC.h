#ifndef BC_20200910_H
#define BC_20200910_H

namespace tndm {

enum class BC : int {
    None = 0,
    Natural = 1,
    Fault = 3,
    Dirichlet = 5,
    Traction = 7,
    FreeSlip = 9,
    Unknown = -1
};

// TODO: Choose these ranges as decided

inline constexpr long DIRICHLET_MIN = 5;
inline constexpr long DIRICHLET_MAX = 5;
inline constexpr long DIRICHLET_DEFAULT = 5;

inline constexpr long FAULT_MIN = 10;
inline constexpr long FAULT_MAX = 1000;
inline constexpr long FAULT_DEFAULT = 3;

inline constexpr long NATURAL_MIN = 1;
inline constexpr long NATURAL_MAX = 1;
inline constexpr long NATURAL_DEFAULT = 1;

// Fixed boundary-condition tags for now
inline constexpr long TRACTION_DEFAULT = 7;
inline constexpr long FREE_SLIP_DEFAULT = 9;

inline constexpr long NO_FACET_TAG = -1;

inline BC classifyBC(long int tag) {

    if (tag == 0)
        return BC::None;

    if (tag == DIRICHLET_DEFAULT || (tag >= DIRICHLET_MIN && tag <= DIRICHLET_MAX))
        return BC::Dirichlet;

    if (tag == FAULT_DEFAULT || (tag >= FAULT_MIN && tag <= FAULT_MAX))
        return BC::Fault;

    if (tag == NATURAL_DEFAULT || (tag >= NATURAL_MIN && tag <= NATURAL_MAX))
        return BC::Natural;

    if (tag == TRACTION_DEFAULT)
        return BC::Traction;

    if (tag == FREE_SLIP_DEFAULT)
        return BC::FreeSlip;

    return BC::Unknown;
}

} // namespace tndm

#endif // BC_20200910_H