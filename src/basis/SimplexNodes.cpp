#include "SimplexNodes.h"
#include "Functions.h"
#include "Quadrature.h"
#include "util/Combinatorics.h"

#include <algorithm>
#include <iterator>
#define _USE_MATH_DEFINES
#include <cmath>

namespace tndm {

/**
 * Port of
 * https://github.com/tcew/nodal-dg/blob/master/Codes1.1/Codes2D/Nodes2D.m
 */
template <> std::vector<std::array<double, 2>> simplexNodes(unsigned degree, double alpha) {
    assert(degree > 0);

    const double alphaOpt[] = {0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 1.2832,
                               1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258};
    if (alpha < 0.0) {
        alpha = (degree < 15) ? alphaOpt[degree - 1] : 5.0 / 3.0;
    }

    unsigned numNodes = binom(degree + 2, 2);
    std::vector<std::array<double, 2>> result(numNodes);
    Eigen::Map<Eigen::MatrixXd> resultMap(result.data()->data(), 2, numNodes);

    Eigen::VectorXd L1(numNodes);
    Eigen::VectorXd L2(numNodes);
    Eigen::VectorXd L3(numNodes);

    std::size_t idx = 0;
    for (auto i : AllIntegerSums<2>(degree)) {
        L1(idx) = i[1] / static_cast<double>(degree);
        L3(idx) = i[0] / static_cast<double>(degree);
        ++idx;
    }
    L2 = Eigen::VectorXd::Ones(numNodes) - L1 - L3;
    resultMap.row(0) = -L2 + L3;
    resultMap.row(1) = (-L2 - L3 + 2.0 * L1) / sqrt(3.0);

    auto blend1 = 4.0 * L2.cwiseProduct(L3);
    auto blend2 = 4.0 * L1.cwiseProduct(L3);
    auto blend3 = 4.0 * L1.cwiseProduct(L2);

    auto factor = [&numNodes, &alpha](auto& blend, auto& L) {
        return blend.cwiseProduct(Eigen::VectorXd::Ones(numNodes) +
                                  (alpha * alpha) * L.cwiseProduct(L));
    };

    Warpfactor warpfactor(degree);
    auto warp1 = warpfactor(L3 - L2);
    auto warp2 = warpfactor(L1 - L3);
    auto warp3 = warpfactor(L2 - L1);
    warp1 = warp1.cwiseProduct(factor(blend1, L1));
    warp2 = warp2.cwiseProduct(factor(blend2, L2));
    warp3 = warp3.cwiseProduct(factor(blend3, L3));

    resultMap.row(0) += warp1 + cos(2.0 * M_PI / 3.0) * warp2 + cos(4.0 * M_PI / 3.0) * warp3;
    resultMap.row(1) += sin(2.0 * M_PI / 3.0) * warp2 + sin(4.0 * M_PI / 3.0) * warp3;

    // to reference triangle
    L1 = (sqrt(3.0) * resultMap.row(1) + Eigen::MatrixXd::Ones(1, numNodes)) / 3.0;
    L2 = (-3.0 * resultMap.row(0) - sqrt(3.0) * resultMap.row(1) +
          Eigen::MatrixXd::Constant(1, numNodes, 2.0)) /
         6.0;
    L3 = (3.0 * resultMap.row(0) - sqrt(3.0) * resultMap.row(1) +
          Eigen::MatrixXd::Constant(1, numNodes, 2.0)) /
         6.0;

    resultMap.row(0) = 0.5 * (-L2 + L3 - L1 + Eigen::VectorXd::Ones(numNodes));
    resultMap.row(1) = 0.5 * (-L2 - L3 + L1 + Eigen::VectorXd::Ones(numNodes));

    return result;
}

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
Eigen::MatrixXd Vandermonde(unsigned degree, std::vector<std::array<double, D>> const& points) {
    assert(binom(degree + D, D) == points.size());

    Eigen::MatrixXd vandermonde(points.size(), binom(degree + D, D));

    for (std::size_t i = 0; i < points.size(); ++i) {
        std::size_t bf = 0;
        for (auto j : AllIntegerSums<D>(degree)) {
            vandermonde(i, bf++) = DubinerP(j, points[i]);
        }
    }

    return vandermonde;
}

Warpfactor::Warpfactor(unsigned N) : diff(N + 1) {
    std::vector<std::array<double, 1>> eqPoints(N + 1);
    for (unsigned i = 0; i < N + 1; ++i) {
        eqPoints[i][0] = 2 * i / static_cast<double>(N) - 1.0;
    }
    auto vandermonde = Vandermonde(N, eqPoints);
    LQR = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(vandermonde.transpose());
    auto glPoints = LegendreGaussLobattoPoints(N + 1, 0, 0);
    for (unsigned i = 0; i < N + 1; ++i) {
        diff[i] = glPoints[i] - eqPoints[i][0];
    }
}

Eigen::VectorXd Warpfactor::operator()(Eigen::VectorXd const& r) {
    const auto Np1 = diff.rows();
    Eigen::MatrixXd phi(Np1, r.size());
    for (unsigned i = 0; i < Np1; ++i) {
        for (unsigned j = 0; j < r.rows(); ++j) {
            phi(i, j) = JacobiP(i, 0, 0, r(j));
        }
    }
    Eigen::MatrixXd L = LQR.solve(phi);
    Eigen::VectorXd warp = L.transpose() * diff;
    assert(warp.size() == r.size());
    for (unsigned i = 0; i < warp.size(); ++i) {
        if (std::fabs(r[i]) < 1.0 - std::numeric_limits<double>::epsilon()) {
            warp[i] /= 1.0 - r[i] * r[i];
        }
    }
    return warp;
}

template Eigen::MatrixXd Vandermonde<2u>(unsigned, std::vector<std::array<double, 2u>> const&);
template Eigen::MatrixXd Vandermonde<3u>(unsigned, std::vector<std::array<double, 3u>> const&);

} // namespace tndm
