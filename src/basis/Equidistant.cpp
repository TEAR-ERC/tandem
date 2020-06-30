#include "Equidistant.h"
#include "mesh/Simplex.h"
#include "util/Combinatorics.h"

#include <iostream>

namespace tndm {

template <std::size_t D>
std::vector<std::array<double, D>> EquidistantNodesFactory<D>::operator()(unsigned degree) const {
    unsigned numNodes = binom(degree + D, D);
    std::vector<std::array<double, D>> result(numNodes);
    // TODO: Generic implementation
    if constexpr (D == 2) {
        if (degree == 1) {
            result = std::vector<std::array<double, 2>>{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
        } else if (degree == 2) {
            result = std::vector<std::array<double, 2>>{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},
                                                        {0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}};
        }
    }
    return result;
}

template class EquidistantNodesFactory<2u>;
template class EquidistantNodesFactory<3u>;

} // namespace tndm
