#ifndef ADAPTIVEOUTPUTSTRATEGY_20201119_H
#define ADAPTIVEOUTPUTSTRATEGY_20201119_H

#include <algorithm>
#include <cmath>
#include <limits>

namespace tndm {

class AdaptiveOutputInterval {
public:
    AdaptiveOutputInterval(double atol, double rtol, double t_min, double t_max)
        : atol_(atol), rtol_(rtol), t_min_(t_min), t_max_(t_max) {}

    bool operator()(double delta_time, double last_VMax, double VMax) const {
        double tol = atol_ + std::max(last_VMax, VMax) * rtol_;
        double lte = std::abs(VMax - last_VMax) / tol;
        return (delta_time > t_max_) || ((delta_time > t_min_) && (lte >= 1.0));
    }

private:
    const double atol_;
    const double rtol_;
    const double t_min_;
    const double t_max_;
};

} // namespace tndm

#endif // ADAPTIVEOUTPUTSTRATEGY_20201119_H
