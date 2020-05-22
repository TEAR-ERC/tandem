#include "GenMesh.h"
#include <utility>

namespace tndm {

template <>
std::array<GenMesh<1>::simplex_t, 1> GenMesh<1>::tessellate(std::array<uint64_t, 2> const& vertGIDs,
                                                            bool) {
    std::array<simplex_t,1> result;
    result[0] = vertGIDs;
    return result;
}

template <>
std::array<GenMesh<2>::simplex_t, 2> GenMesh<2>::tessellate(std::array<uint64_t, 4> const& vertGIDs,
                                                            bool isOdd) {
    std::array<simplex_t,2> result;
    result[0] = {vertGIDs[0], vertGIDs[1], vertGIDs[2]};
    result[1] = {vertGIDs[1], vertGIDs[3], vertGIDs[2]};
    return result;
}

template <>
std::array<GenMesh<3>::simplex_t, 5> GenMesh<3>::tessellate(std::array<uint64_t, 8> const& vertGIDs,
                                                            bool isOdd) {
    std::array<simplex_t,5> result;
    if (isOdd) {
        result[0] = {vertGIDs[0], vertGIDs[1], vertGIDs[2], vertGIDs[4]};
        result[1] = {vertGIDs[1], vertGIDs[2], vertGIDs[3], vertGIDs[7]};
        result[2] = {vertGIDs[2], vertGIDs[4], vertGIDs[6], vertGIDs[7]};
        result[3] = {vertGIDs[1], vertGIDs[4], vertGIDs[5], vertGIDs[7]};
        result[4] = {vertGIDs[2], vertGIDs[4], vertGIDs[1], vertGIDs[7]};
    } else {
        result[0] = {vertGIDs[0], vertGIDs[2], vertGIDs[3], vertGIDs[6]};
        result[1] = {vertGIDs[0], vertGIDs[1], vertGIDs[3], vertGIDs[5]};
        result[2] = {vertGIDs[0], vertGIDs[4], vertGIDs[5], vertGIDs[6]};
        result[3] = {vertGIDs[3], vertGIDs[5], vertGIDs[6], vertGIDs[7]};
        result[4] = {vertGIDs[0], vertGIDs[3], vertGIDs[5], vertGIDs[6]};
    }
    return result;
}
}


