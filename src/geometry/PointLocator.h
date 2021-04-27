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
    std::array<double, D> x;
};

template <std::size_t D> class PointLocator {
public:
    using point_t = std::array<double, D>;

    PointLocator(std::shared_ptr<Curvilinear<D>> cl, double atol = 1e-10, double rtol = 1e-9,
                 double bbox_factor = 1.5, double max_its = 100);

    template <typename Iterator>
    auto locate(point_t const& point, Iterator first, Iterator last) const
        -> PointLocatorResult<D> {
        auto r = PointLocatorResult<D>{
            std::numeric_limits<std::size_t>::max(), std::numeric_limits<double>::max(), {}, {}};

        auto closest_point_finder = SimplexDistance(Simplex<D>::referenceSimplexVertices());
        auto tmp = XInvTmp(*cl_);

        while (first != last) {
            if (inside_bbox(*first, point)) {
                auto xi = xInv(*first, point, tmp);
                auto [dist_local, xi_c, bary] = closest_point_finder.closest(xi);

                auto E = cl_->evaluateBasisAt({xi_c});
                cl_->map(*first, E, tmp.F);
                point_t x{};
                for (std::size_t d = 0; d < D; ++d) {
                    x[d] = tmp.F(d, 0);
                }

                double dist = norm(x - point);
                if (dist < r.dist) {
                    r.dist = dist;
                    r.no = *first;
                    r.xi = xi_c;
                    r.x = x;
                }
            }
            ++first;
        }

        return r;
    }

    auto locate(point_t const& point) const {
        auto range = Range<std::size_t>(0, cl_->numElements());
        return locate(point, range.begin(), range.end());
    }

    Curvilinear<D> const& cl() const { return *cl_; }

private:
    bool inside_bbox(std::size_t elNo, point_t const& point) const;

    struct XInvTmp {
        XInvTmp(Curvilinear<D> const& cl) : F(cl.mapResultInfo(1)), J(cl.jacobianResultInfo(1)) {}
        Managed<Matrix<double>> F;
        Managed<Tensor<double, 3u>> J;
    };
    auto xInv(std::size_t elNo, point_t const& point, XInvTmp& tmp) const -> point_t;

    std::shared_ptr<Curvilinear<D>> cl_;
    double atol_, rtol_;
    std::size_t max_its_;

    std::vector<std::array<std::array<double, 2>, D>> bbox_;
};

template <std::size_t D> struct BoundaryPointLocatorResult {
    std::size_t no;
    double dist;
    std::array<double, D - 1> chi;
    std::array<double, D> x;
};

template <std::size_t D> class BoundaryPointLocator {
public:
    using point_t = std::array<double, D>;

    BoundaryPointLocator(std::shared_ptr<PointLocator<D>> pl, LocalSimplexMesh<D> const& mesh,
                         std::vector<std::size_t> const& fctNos);

    auto locate(point_t const& point) const -> BoundaryPointLocatorResult<D>;

private:
    std::shared_ptr<PointLocator<D>> pl_;
    std::unordered_set<std::size_t> elNos_;
    std::unordered_multimap<std::size_t, std::pair<std::size_t, int>> bnds_;
    std::array<SimplexDistance<D, D>, D + 1> closest_;
};

} // namespace tndm

#endif // POINTLOCATOR_20210415_H
