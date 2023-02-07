#ifndef FACETINFO_20200910_H
#define FACETINFO_20200910_H

#include <array>
#include <cstddef>

namespace tndm {

struct SideInfo {
    std::size_t fctNo;
    int side;
    std::size_t lid;
    std::size_t localNo;
    int ptag;
    int etag;
};

struct FacetInfo {
    std::array<bool, 2> inside;
    std::array<std::size_t, 2> up;
    std::array<std::size_t, 2> g_up;
    std::array<std::size_t, 2> localNo;
    int ptag;
    int etag;
};

struct ElementInfo {
    int ptag;
    int etag;
};

} // namespace tndm

#endif // FACETINFO_20200910_H
