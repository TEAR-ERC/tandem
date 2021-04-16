#include "PointLocator.h"

#include "geometry/SimplexDistance.h"
#include "tensor/EigenMap.h"

#include <Eigen/LU>
#include <utility>

namespace tndm {

template <std::size_t D>
auto PointLocator<D>::locate(point_t const& point) const -> std::pair<std::size_t, point_t> {
    std::size_t closest_elNo = std::numeric_limits<std::size_t>::max();
    double min_dist = std::numeric_limits<double>::max();
    point_t closest_xi{};

    auto closest_point_finder = SimplexDistance<D>(Simplex<D>::referenceSimplexVertices());

    auto F_Q = Managed(cl_->mapResultInfo(1));
    auto J_Q = Managed(cl_->jacobianResultInfo(1));
    for (std::size_t elNo = 0, num = cl_->numElements(); elNo < num; ++elNo) {
        auto xi = point;
        std::size_t it = 0;
        while (it < max_its_) {
            // point - X(xi)
            auto E = cl_->evaluateBasisAt({xi});
            cl_->map(elNo, E, F_Q);
            auto X_0 = F_Q.subtensor(slice{}, 0);
            auto F = EigenMap(X_0);
            double residual = 0.0;
            for (std::size_t d = 0; d < D; ++d) {
                F(d) = point[d] - F(d);
                residual = F(d) * F(d);
            }
            if (residual < tol_) {
                break;
            }

            // JX(xi)
            auto gradE = cl_->evaluateGradientAt({xi});
            cl_->jacobian(elNo, gradE, J_Q);
            auto J_0 = J_Q.subtensor(slice{}, slice{}, 0);
            auto J = EigenMap(J_0);

            Eigen::Matrix<double, D, 1> h = J.partialPivLu().solve(F);
            for (std::size_t d = 0; d < D; ++d) {
                xi[d] += h(d);
            }

            ++it;
        }
        auto [dist, xi_c] = closest_point_finder.closest(xi);
        if (dist < min_dist) {
            min_dist = dist;
            closest_elNo = elNo;
            closest_xi = xi_c;
        }
    }
    return std::make_pair(closest_elNo, closest_xi);
}

template class PointLocator<1ul>;
template class PointLocator<2ul>;
template class PointLocator<3ul>;

} // namespace tndm
