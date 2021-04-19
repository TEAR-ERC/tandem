#ifndef POINTLOCATOR_20210415_H
#define POINTLOCATOR_20210415_H

#include "geometry/Curvilinear.h"
#include "geometry/SimplexDistance.h"
#include "mesh/LocalSimplexMesh.h"
#include "util/Range.h"

#include <array>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tndm {

template <std::size_t D> struct PointLocatorResult {
    std::size_t no;
    double dist;
    std::array<double, D> xi;
};

template <std::size_t D> class PointLocator {
public:
    using point_t = std::array<double, D>;

    PointLocator(std::shared_ptr<Curvilinear<D>> cl, double tol = 0.0, double max_its = 100)
        : cl_(std::move(cl)), tol_(tol), max_its_(max_its) {}

    template <typename Iterator>
    auto locate(point_t const& point, Iterator first, Iterator last) const
        -> PointLocatorResult<D> {
        auto r = PointLocatorResult<D>{
            std::numeric_limits<std::size_t>::max(), std::numeric_limits<double>::max(), {}};

        auto closest_point_finder = SimplexDistance(Simplex<D>::referenceSimplexVertices());
        auto tmp = XInvTmp(*cl_);

        while (first != last) {
            auto xi = xInv(*first, point, tmp);
            auto [dist, xi_c, bary] = closest_point_finder.closest(xi);
            if (dist < r.dist) {
                r.dist = dist;
                r.no = *first;
                r.xi = xi_c;
            }
            ++first;
        }

        return r;
    }

    auto locate(point_t const& point) const {
        auto range = Range<std::size_t>(0, cl_->numElements());
        return locate(point, range.begin(), range.end());
    }

private:
    struct XInvTmp {
        XInvTmp(Curvilinear<D> const& cl) : F(cl.mapResultInfo(1)), J(cl.jacobianResultInfo(1)) {}
        Managed<Matrix<double>> F;
        Managed<Tensor<double, 3u>> J;
    };
    auto xInv(std::size_t elNo, point_t const& point, XInvTmp& tmp) const -> point_t;

    std::shared_ptr<Curvilinear<D>> cl_;
    double tol_ = 10.0 * std::numeric_limits<double>::epsilon();
    std::size_t max_its_;
};

template <std::size_t D> class BoundaryPointLocator {
public:
    using point_t = std::array<double, D>;

    BoundaryPointLocator(std::shared_ptr<PointLocator<D>> pl, LocalSimplexMesh<D> const& mesh,
                         std::vector<std::size_t> const& fctNos);

    auto locate(point_t const& point) const -> PointLocatorResult<D - 1>;

private:
    std::shared_ptr<PointLocator<D>> pl_;
    std::unordered_set<std::size_t> elNos_;
    std::unordered_multimap<std::size_t, std::pair<std::size_t, int>> bnds_;
    std::array<SimplexDistance<D, D>, D + 1> closest_;
};

} // namespace tndm

#endif // POINTLOCATOR_20210415_H
