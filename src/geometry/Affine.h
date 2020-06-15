#ifndef GEOMETRY_AFFINE_H
#define GEOMETRY_AFFINE_H

#include "Vector.h"

#include <Eigen/Core>

#include <array>
#include <numeric>

namespace tndm {

template <std::size_t D> class RefPlexToGeneralPlex {
public:
    RefPlexToGeneralPlex(std::array<std::array<double, D>, D + 1> const& generalPlexVerts)
        : verts(generalPlexVerts) {}

    [[nodiscard]] auto operator()(std::array<double, D> const& xi) const {
        std::array<double, D> x;
        x = (1.0 - std::accumulate(xi.begin(), xi.end(), 0.0)) * verts[0];
        for (std::size_t d = 0; d < D; ++d) {
            x = x + xi[d] * verts[d + 1];
        }
        return x;
    }

private:
    std::array<std::array<double, D>, D + 1> verts;
};

template <std::size_t D> class GeneralPlexToRefPlex {
public:
    GeneralPlexToRefPlex(std::array<std::array<double, D>, D + 1> const& generalPlexVerts) {
        Eigen::Matrix<double, D, D> A;
        for (std::size_t i = 0; i < D; ++i) {
            for (std::size_t j = 0; j < D; ++j) {
                A(i, j) = generalPlexVerts[j + 1][i] - generalPlexVerts[0][i];
            }
            x0[i] = generalPlexVerts[0][i];
        }
        AQR = decltype(AQR)(A);
    }

    [[nodiscard]] auto operator()(std::array<double, D> const& x) const {
        std::array<double, D> xi;
        auto diff = x - x0;
        Eigen::Map<Eigen::Matrix<double, D, 1>>(xi.data()) =
            AQR.solve(Eigen::Map<Eigen::Matrix<double, D, 1>>(diff.data()));
        return xi;
    }

private:
    std::array<double, D> x0;
    Eigen::ColPivHouseholderQR<Eigen::Matrix<double, D, D>> AQR;
};

} // namespace tndm

#endif // GEOMETRY_AFFINE_H
