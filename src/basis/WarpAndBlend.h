#ifndef WARPANDBLEND_H
#define WARPANDBLEND_H

#include <Eigen/Dense>

#include <array>
#include <vector>

namespace tndm {

/**
 * @brief Returns warp & blend nodes.
 *
 * From Hesthaven and Warburton, "Nodal Discontinuous Galerkin Methods", Springer, 2008.
 *
 * @param degree Maximum polynomial degree
 * @param alpha See Book. Passing -1.0 automatically employs the optimal value until N=14
 */
template <std::size_t D>
std::vector<std::array<double, D>> warpAndBlendNodes(unsigned degree, double alpha = -1.0);

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

} // namespace tndm

#endif // WARPANDBLEND_H
