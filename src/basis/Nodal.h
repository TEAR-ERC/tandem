#ifndef BASIS_NODAL_H
#define BASIS_NODAL_H

#include <array>
#include <vector>

#include <Eigen/Core>

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

template <std::size_t D> class LebesgueFunction {
public:
    LebesgueFunction(unsigned degree, std::vector<std::array<double, D>> const& nodes);
    double operator()(std::array<double, D> const& xi);

private:
    unsigned degree;
    Eigen::MatrixXd vInvT;
    Eigen::VectorXd phi, L;
};

template <std::size_t D> class NodesFactory {
public:
    virtual std::vector<std::array<double, D>> operator()(unsigned degree) const = 0;
};

} // namespace tndm

#endif // BASIS_NODAL_H
