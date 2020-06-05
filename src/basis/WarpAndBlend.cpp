#include "WarpAndBlend.h"
#include "Functions.h"
#include "geometry/Affine.h"
#include "mesh/Simplex.h"
#include "util/Combinatorics.h"

using Eigen::ColPivHouseholderQR;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::RowMajor;
using Eigen::VectorXd;

namespace tndm {

double warpAndBlendAlpha(std::size_t D, unsigned degree) {
    assert(D == 2 || D == 3);
    assert(degree > 0);
    constexpr double alphaOpt2[] = {
        0.0000,  0.0000,  1.4152,  0.1001,  0.2751,  0.9800,  1.0999,  1.2832,  1.3648,  1.4773,
        1.4959,  1.5743,  1.5770,  1.6223,  1.6258,  1.64831, 1.76303, 1.92727, 1.92962, 1.91251,
        1.91289, 1.89479, 1.89255, 1.88016, 1.87555, 1.86613, 1.86106, 1.85368, 1.84874, 1.84319};

    constexpr double alphaOpt3[] = {0.0,    0.0,     0.0,    0.1002, 1.1332, 1.5608, 1.3413, 1.2577,
                                    1.1603, 1.10153, 0.6080, 0.4523, 0.8856, 0.8717, 0.9655};

    const auto tabAlpha = [&degree](const double* alphaOpt, unsigned numEntries) {
        return (degree <= numEntries) ? alphaOpt[degree - 1u] : 5.0 / 3.0;
    };

    return (D == 2) ? tabAlpha(alphaOpt2, sizeof(alphaOpt2) / sizeof(double))
                    : tabAlpha(alphaOpt3, sizeof(alphaOpt3) / sizeof(double));
}

template <>
std::vector<std::array<double, 2>> WarpAndBlendFactory<2>::operator()(unsigned degree) const {
    assert(degree > 0);

    double alpha = alphaFun(degree);

    unsigned numNodes = binom(degree + 2, 2);
    std::vector<std::array<double, 2>> result(numNodes);
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> resultMap(result.data()->data(), numNodes, 2);

    MatrixXd L(numNodes, 3);

    std::size_t idx = 0;
    for (auto i : AllIntegerSums<2>(degree)) {
        L(idx, 2) = i[1] / static_cast<double>(degree);
        L(idx, 1) = i[0] / static_cast<double>(degree);
        L(idx, 0) = 1.0 - L(idx, 1) - L(idx, 2);
        ++idx;
    }

    auto equilateralVerts = equilateralTriangle();
    const Map<const Matrix<double, 3, 2, RowMajor>> equiMap(equilateralVerts.data()->data());

    resultMap = L * equiMap + warpAndBlendTriangle(degree, alpha, L);

    GeneralPlexToRefPlex equiToRef(equilateralVerts);
    for (std::size_t n = 0; n < numNodes; ++n) {
        result[n] = equiToRef(result[n]);
    }

    return result;
}

template <>
std::vector<std::array<double, 3>> WarpAndBlendFactory<3>::operator()(unsigned degree) const {
    assert(degree > 0);

    double alpha = alphaFun(degree);

    unsigned numNodes = binom(degree + 3, 3);
    std::vector<std::array<double, 3>> result(numNodes);
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> resultMap(result.data()->data(), numNodes, 3);

    MatrixXd L(numNodes, 4);
    std::size_t idx = 0;
    for (auto i : AllIntegerSums<3>(degree)) {
        L(idx, 3) = i[2] / static_cast<double>(degree);
        L(idx, 2) = i[1] / static_cast<double>(degree);
        L(idx, 1) = i[0] / static_cast<double>(degree);
        L(idx, 0) = 1.0 - L(idx, 1) - L(idx, 2) - L(idx, 3);
        ++idx;
    }

    auto eqVerts = equilateralTetrahedron();
    const Map<const Matrix<double, 4, 3, RowMajor>> equiMap(eqVerts.data()->data());

    resultMap = MatrixXd::Zero(numNodes, 3);

    auto refTet = Simplex<3>::referenceSimplex();
    MatrixXd Lface(numNodes, 3);
    MatrixXd warp(numNodes, 3);
    VectorXd blend(numNodes);
    VectorXd denom(numNodes);
    VectorXd mask(numNodes);
    VectorXd factor(numNodes);
    for (auto const& facet : refTet.downward()) {
        Matrix<double, 2, 3> tangents;
        tangents.row(0) = (equiMap.row(facet[1]) - equiMap.row(facet[0])).normalized();
        tangents.row(1) =
            (equiMap.row(facet[2]) - 0.5 * (equiMap.row(facet[1]) + equiMap.row(facet[0])))
                .normalized();

        Lface << L.col(facet[0]), L.col(facet[1]), L.col(facet[2]);
        warp = warpAndBlendTriangle(degree, alpha, Lface) * tangents;

        std::vector<uint64_t> mv;
        std::set_difference(refTet.begin(), refTet.end(), facet.begin(), facet.end(),
                            std::inserter(mv, mv.begin()));
        assert(mv.size() == 1);

        blend = Lface.col(0);
        for (unsigned i = 1; i < 3; ++i) {
            blend = blend.cwiseProduct(Lface.col(i));
        }
        denom = (Lface.col(0) + 0.5 * L.col(mv[0]));
        for (unsigned i = 1; i < 3; ++i) {
            denom = denom.cwiseProduct(Lface.col(i) + 0.5 * L.col(mv[0]));
        }
        double alpha2 = alpha * alpha;
        factor = VectorXd::Ones(numNodes) + alpha2 * L.col(mv[0]).cwiseProduct(L.col(mv[0]));

        for (Eigen::Index i = 0; i < numNodes; ++i) {
            if (std::fabs(denom(i)) < std::numeric_limits<double>::epsilon()) {
                blend(i) = 1.0;
                mask(i) = 0.0;
            } else {
                blend(i) = factor(i) * blend(i) / denom(i);
                mask(i) = 1.0;
            }
        }

        resultMap = mask.asDiagonal() * resultMap + blend.asDiagonal() * warp;
    }

    resultMap += L * equiMap;

    GeneralPlexToRefPlex equiToRef(eqVerts);
    for (std::size_t n = 0; n < numNodes; ++n) {
        result[n] = equiToRef(result[n]);
    }

    return result;
}

Warpfactor::Warpfactor(unsigned N) : diff(N + 1) {
    std::vector<std::array<double, 1>> eqPoints(N + 1);
    for (unsigned i = 0; i < N + 1; ++i) {
        eqPoints[i][0] = 2 * i / static_cast<double>(N) - 1.0;
    }
    auto vandermonde = Vandermonde(N, eqPoints);
    LQR = ColPivHouseholderQR<MatrixXd>(vandermonde.transpose());
    auto glPoints = LegendreGaussLobattoPoints(N + 1, 0, 0);
    for (unsigned i = 0; i < N + 1; ++i) {
        diff[i] = glPoints[i] - eqPoints[i][0];
    }
}

VectorXd Warpfactor::operator()(VectorXd const& r) {
    const auto Np1 = diff.rows();
    MatrixXd phi(Np1, r.size());
    for (unsigned i = 0; i < Np1; ++i) {
        for (unsigned j = 0; j < r.rows(); ++j) {
            phi(i, j) = JacobiP(i, 0, 0, r(j));
        }
    }
    MatrixXd L = LQR.solve(phi);
    VectorXd warp = L.transpose() * diff;
    assert(warp.size() == r.size());
    for (unsigned i = 0; i < warp.size(); ++i) {
        if (std::fabs(r[i]) < 1.0 - std::numeric_limits<double>::epsilon()) {
            warp[i] /= 1.0 - r[i] * r[i];
        }
    }
    return warp;
}

Eigen::MatrixXd warpAndBlendTriangle(unsigned degree, double alpha, Eigen::MatrixXd const& L) {
    const auto numNodes = L.rows();
    const auto equilateralVerts = equilateralTriangle();
    const auto refTri = Simplex<2>::referenceSimplex();

    Warpfactor warpfactor(degree);
    MatrixXd result = MatrixXd::Zero(numNodes, 2);
    VectorXd warp(numNodes);
    VectorXd blend(numNodes);

    for (auto const& facet : refTri.downward()) {
        warp = warpfactor(L.col(facet[1]) - L.col(facet[0]));

        blend = 4.0 * L.col(facet[0]).cwiseProduct(L.col(facet[1]));

        std::vector<uint64_t> mv;
        std::set_difference(refTri.begin(), refTri.end(), facet.begin(), facet.end(),
                            std::inserter(mv, mv.begin()));
        assert(mv.size() == 1);
        blend = blend.cwiseProduct(VectorXd::Ones(numNodes) +
                                   (alpha * alpha) * L.col(mv[0]).cwiseProduct(L.col(mv[0])));

        auto tangent = normalize(equilateralVerts[facet[1]] - equilateralVerts[facet[0]]);
        result += warp.cwiseProduct(blend) * Map<Matrix<double, 1, 2>>(tangent.data());
    }

    return result;
}

} // namespace tndm
