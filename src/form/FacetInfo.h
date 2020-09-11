#ifndef FACETINFO_20200910_H
#define FACETINFO_20200910_H

#include "form/BC.h"

#include <array>
#include <cstddef>

namespace tndm {

struct FacetInfo {
    std::array<bool, 2> inside;
    std::array<std::size_t, 2> up;
    std::array<std::size_t, 2> g_up;
    std::array<std::size_t, 2> localNo;
    BC bc;
};
} // namespace tndm

#endif // FACETINFO_20200910_H
