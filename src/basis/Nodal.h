#ifndef BASIS_NODAL_H
#define BASIS_NODAL_H

#include <array>
#include <vector>

#include <Eigen/Dense>

namespace tndm {

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

} // namespace tndm

#endif // BASIS_NODAL_H
