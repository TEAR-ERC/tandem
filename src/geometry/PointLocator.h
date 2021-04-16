#ifndef POINTLOCATOR_20210415_H
#define POINTLOCATOR_20210415_H

#include "geometry/Curvilinear.h"
#include "util/Range.h"

#include <array>
#include <limits>
#include <memory>
#include <utility>

namespace tndm {

template <std::size_t D> class PointLocator {
public:
    using point_t = std::array<double, D>;

    PointLocator(std::shared_ptr<Curvilinear<D>> cl, double tol = 0.0, double max_its = 100)
        : cl_(std::move(cl)), tol_(tol), max_its_(max_its) {}

    auto locate(point_t const& point) const -> std::pair<std::size_t, point_t>;

private:
    std::shared_ptr<Curvilinear<D>> cl_;
    double tol_ = 10.0 * std::numeric_limits<double>::epsilon();
    std::size_t max_its_;
};

} // namespace tndm

#endif // POINTLOCATOR_20210415_H
