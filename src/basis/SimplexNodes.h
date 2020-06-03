#ifndef SIMPLEXNODES_H
#define SIMPLEXNODES_H

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
std::vector<std::array<double, D>> simplexNodes(unsigned degree, double alpha = -1.0);

/**
 * @brief Returns Legendre-Gauss-Lobatto points on [-1,1]
 *
 * @param n Number of points with n >= 2
 * @param a
 * @param b
 */
std::vector<double> LegendreGaussLobattoPoints(unsigned n, unsigned a, unsigned b);

/**
 * @brief Compute generalized Vandermonde matrix.
 */
template <std::size_t D>
Eigen::MatrixXd Vandermonde(unsigned degree, std::vector<std::array<double, D>> const& points);

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

#endif // SIMPLEXNODES_H
