#ifndef ADAPTIVEOUTPUTSTRATEGY_20201119_H
#define ADAPTIVEOUTPUTSTRATEGY_20201119_H

#include <algorithm>
#include <cmath>
#include <limits>

namespace tndm {

enum class AdaptiveOutputStrategy { Threshold, Exponential, Unknown };

class AdaptiveOutputInterval {
public:
    AdaptiveOutputInterval(double V_ref, double t_min, double t_max,
                           AdaptiveOutputStrategy strategy)
        : V_ref_(V_ref), t_min_(t_min), t_max_(t_max), strategy_(strategy) {}

    double operator()(double VMax) const {
        double interval = 0.0;
        switch (strategy_) {
        case AdaptiveOutputStrategy::Threshold:
            interval = VMax >= V_ref_ ? t_min_ : t_max_;
            break;
        case AdaptiveOutputStrategy::Exponential: {
            double falloff = log(t_min_ / t_max_);
            VMax = std::min(V_ref_, VMax);
            interval = t_max_ * exp(falloff * VMax / V_ref_);
            break;
        }
        default:
            break;
        }
        return interval;
    }

    double V_ref_;
    double t_min_;
    double t_max_;
    AdaptiveOutputStrategy strategy_;
};

} // namespace tndm

#endif // ADAPTIVEOUTPUTSTRATEGY_20201119_H
