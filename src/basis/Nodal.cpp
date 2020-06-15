#include "Nodal.h"
#include "Functions.h"
#include "quadrules/GaussJacobi.h"
#include "util/Combinatorics.h"

#include <algorithm>
#include <iterator>

using Eigen::MatrixXd;
using Eigen::VectorXd;

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

template <std::size_t D>
LebesgueFunction<D>::LebesgueFunction(unsigned degree,
                                      std::vector<std::array<double, D>> const& nodes)
    : degree(degree), phi(nodes.size()), L(nodes.size()) {
    auto vandermonde = Vandermonde(degree, nodes);
    vInvT = vandermonde.transpose().inverse();
}

template <std::size_t D> double LebesgueFunction<D>::operator()(std::array<double, D> const& xi) {
    std::size_t bf = 0;
    for (auto j : AllIntegerSums<D>(degree)) {
        phi(bf++) = DubinerP(j, xi);
    }
    assert(bf == vInvT.cols());
    L = vInvT * phi;
    return L.lpNorm<1>();
}

template MatrixXd Vandermonde<1u>(unsigned, std::vector<std::array<double, 1u>> const&);
template MatrixXd Vandermonde<2u>(unsigned, std::vector<std::array<double, 2u>> const&);
template MatrixXd Vandermonde<3u>(unsigned, std::vector<std::array<double, 3u>> const&);

template class LebesgueFunction<1u>;
template class LebesgueFunction<2u>;
template class LebesgueFunction<3u>;

} // namespace tndm
