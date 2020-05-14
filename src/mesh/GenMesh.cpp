#include "GenMesh.h"
#include <utility>

namespace tndm {

template<>
std::array<Simplex<1>,1> tessellate(std::array<int,2> const& corners, bool) {
    std::array<Simplex<1>,1> result;
    result[0] = Simplex<1>(corners);
    return result;
}

template<>
std::array<Simplex<2>,2> tessellate(std::array<int,4> const& corners, bool isOdd) {
    std::array<Simplex<2>,2> result;
    result[0] = Simplex<2>({corners[0], corners[1], corners[2]});
    result[1] = Simplex<2>({corners[1], corners[3], corners[2]});
    return result;
}

template<>
std::array<Simplex<3>,5> tessellate(std::array<int,8> const& corners, bool isOdd) {
    std::array<Simplex<3>,5> result;
    if (isOdd) {
        result[0] = Simplex<3>({corners[0], corners[1], corners[2], corners[4]});
        result[1] = Simplex<3>({corners[1], corners[2], corners[3], corners[7]});
        result[2] = Simplex<3>({corners[2], corners[4], corners[6], corners[7]});
        result[3] = Simplex<3>({corners[1], corners[4], corners[5], corners[7]});
        result[4] = Simplex<3>({corners[2], corners[4], corners[1], corners[7]});
    } else {
        result[0] = Simplex<3>({corners[0], corners[2], corners[3], corners[6]});
        result[1] = Simplex<3>({corners[0], corners[1], corners[3], corners[5]});
        result[2] = Simplex<3>({corners[0], corners[4], corners[5], corners[6]});
        result[3] = Simplex<3>({corners[3], corners[5], corners[6], corners[7]});
        result[4] = Simplex<3>({corners[0], corners[3], corners[5], corners[6]});
    }
    return result;
}

}


