#include "Nodal.h"
#include "Functions.h"
#include "Quadrature.h"
#include "util/Combinatorics.h"

#include <algorithm>
#include <iterator>

using Eigen::MatrixXd;

namespace tndm {

std::vector<double> LegendreGaussLobattoPoints(unsigned n, unsigned a, unsigned b) {
    assert(n >= 2);

    auto gjPoints = GaussJacobi(n - 2, a + 1, b + 1).points();
    std::vector<double> glPoints;
    glPoints.reserve(n);
    glPoints.push_back(-1.0);
    std::copy(gjPoints.rbegin(), gjPoints.rend(), std::back_inserter(glPoints));
    glPoints.push_back(1.0);
    return glPoints;
}

template <std::size_t D>
MatrixXd Vandermonde(unsigned degree, std::vector<std::array<double, D>> const& points) {
    assert(binom(degree + D, D) == points.size());

    MatrixXd vandermonde(points.size(), binom(degree + D, D));

    for (std::size_t i = 0; i < points.size(); ++i) {
        std::size_t bf = 0;
        for (auto j : AllIntegerSums<D>(degree)) {
            vandermonde(i, bf++) = DubinerP(j, points[i]);
        }
    }

    return vandermonde;
}

template MatrixXd Vandermonde<1u>(unsigned, std::vector<std::array<double, 1u>> const&);
template MatrixXd Vandermonde<2u>(unsigned, std::vector<std::array<double, 2u>> const&);
template MatrixXd Vandermonde<3u>(unsigned, std::vector<std::array<double, 3u>> const&);

} // namespace tndm
