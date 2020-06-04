#include "WarpAndBlend.h"
#include "Functions.h"
#include "Nodal.h"
#include "geometry/Affine.h"
#include "util/Combinatorics.h"

using Eigen::ColPivHouseholderQR;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::RowMajor;
using Eigen::VectorXd;

namespace tndm {

template <> std::vector<std::array<double, 2>> warpAndBlendNodes(unsigned degree, double alpha) {
    assert(degree > 0);

    const double alphaOpt[] = {0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 1.2832,
                               1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258};
    if (alpha < 0.0) {
        alpha = (degree < 15) ? alphaOpt[degree - 1] : 5.0 / 3.0;
    }

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

    const std::array<std::array<double, 2>, 3> equilateralVerts{
        {{-1.0, -1.0 / sqrt(3.0)}, {1.0, -1.0 / sqrt(3.0)}, {0.0, 2.0 / sqrt(3.0)}}};
    const Map<const Matrix<double, 3, 2, RowMajor>> equiMap(equilateralVerts.data()->data());

    std::array<std::array<double, 2>, 3> tangents;
    for (std::size_t d = 0; d < 3; ++d) {
        tangents[d] = normalize(equilateralVerts[(d + 1) % 3] - equilateralVerts[d]);
    }
    const Map<const Matrix<double, 3, 2, RowMajor>> tangentMap(tangents.data()->data());

    Warpfactor warpfactor(degree);
    MatrixXd warp(numNodes, 3);
    for (Eigen::Index d = 0; d < warp.cols(); ++d) {
        warp.col(d) = warpfactor(L.col((d + 1) % warp.cols()) - L.col(d));
    }

    MatrixXd blend(numNodes, 3);
    for (Eigen::Index d = 0; d < warp.cols(); ++d) {
        blend.col(d) = 4.0 * L.col(d).cwiseProduct(L.col((d + 1) % warp.cols()));
    }

    auto factor = [&numNodes, &alpha](auto const& blend, auto const& L) {
        return blend.cwiseProduct(VectorXd::Ones(numNodes) + (alpha * alpha) * L.cwiseProduct(L));
    };
    for (Eigen::Index d = 0; d < warp.cols(); ++d) {
        warp.col(d) = warp.col(d).cwiseProduct(factor(blend.col(d), L.col((d + 2) % warp.cols())));
        // warp.col(d) = warp.col(d).cwiseProduct(factor(blend.col(d), L.col(d)));
    }

    resultMap = L * equiMap + warp * tangentMap;

    GeneralPlexToRefPlex equiToRef(equilateralVerts);
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

} // namespace tndm
