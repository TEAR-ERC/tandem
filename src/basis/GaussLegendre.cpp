#include "GaussLegendre.h"
#include "geometry/Affine.h"
#include "quadrules/GaussJacobi.h"
#include "quadrules/IntervalQuadratureRule.h"

#include <cstddef>

namespace tndm {

std::vector<std::array<double, 1u>> GaussLegendreFactory::operator()(unsigned degree) const {
    unsigned numNodes = degree + 1;
    auto gj = GaussJacobi(numNodes, 0, 0);
    std::vector<std::array<double, 1>> result(numNodes);
    for (unsigned i = 0; i < numNodes; ++i) {
        result[i][0] = gj.points()[i];
    }

    std::array<std::array<double, 1>, 2> gjVerts = {{{-1.0}, {1.0}}};
    GeneralPlexToRefPlex equiToRef(gjVerts);
    for (std::size_t n = 0; n < numNodes; ++n) {
        result[n] = equiToRef(result[n]);
    }
    return result;
}

} // namespace tndm
