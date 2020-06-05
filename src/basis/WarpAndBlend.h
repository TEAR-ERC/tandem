#ifndef WARPANDBLEND_H
#define WARPANDBLEND_H

#include "Nodal.h"

#include <Eigen/Dense>

#include <array>
#include <functional>
#include <vector>

namespace tndm {

/**
 * @brief Returns tabulated ("optimal") alpha value.
 *
 * @param D simplex dimension (D = 2 or D = 3)
 * @param degree polynomial degree
 */
double warpAndBlendAlpha(std::size_t D, unsigned degree);

/**
 * @brief Returns warp & blend nodes.
 *
 * From Hesthaven and Warburton, "Nodal Discontinuous Galerkin Methods", Springer, 2008.
 *
 * @param degree Maximum polynomial degree
 * @param alpha See Book.
 */
template <std::size_t D> class WarpAndBlendFactory : public NodesFactory<D> {
public:
    WarpAndBlendFactory()
        : alphaFun([](unsigned degree) { return warpAndBlendAlpha(D, degree); }) {}
    WarpAndBlendFactory(std::function<double(unsigned)> alphaFun) : alphaFun(std::move(alphaFun)) {}
    virtual std::vector<std::array<double, D>> operator()(unsigned degree) const;

private:
    std::function<double(unsigned)> alphaFun;
};

/**
 * @brief Helper class for simplexNodes.
 */
class Warpfactor {
public:
    Warpfactor(unsigned N);
    Eigen::VectorXd operator()(Eigen::VectorXd const& r);

private:
    Eigen::VectorXd diff;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> LQR;
};

inline std::array<std::array<double, 2>, 3> equilateralTriangle() {
    return {{{-1.0, -1.0 / sqrt(3.0)}, {1.0, -1.0 / sqrt(3.0)}, {0.0, 2.0 / sqrt(3.0)}}};
}

inline std::array<std::array<double, 3>, 4> equilateralTetrahedron() {
    return {{{-1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0)},
             {1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0)},
             {0.0, 2.0 / sqrt(3.0), -1.0 / sqrt(6.0)},
             {0.0, 0.0, 3.0 / sqrt(6.0)}}};
}

Eigen::MatrixXd warpAndBlendTriangle(unsigned degree, double alpha, Eigen::MatrixXd const& L);

} // namespace tndm

#endif // WARPANDBLEND_H
