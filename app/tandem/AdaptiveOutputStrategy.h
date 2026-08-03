#ifndef ADAPTIVEOUTPUTSTRATEGY_20201119_H
#define ADAPTIVEOUTPUTSTRATEGY_20201119_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace tndm {

class AdaptiveOutputInterval {
public:
    AdaptiveOutputInterval(double atol, double rtol, double t_min, double t_max,
                           std::optional<double> v_th = std::nullopt, bool high_freq = false)
        : atol_(atol), rtol_(rtol), t_min_(t_min), t_max_(t_max), v_th_(v_th),
          high_freq_(high_freq) {}

    bool operator()(double delta_time, double last_VMax, double VMax) const {
        double tol = atol_ + std::max(last_VMax, VMax) * rtol_;
        double lte = std::abs(VMax - last_VMax) / tol;
        return (delta_time > t_max_) || ((delta_time > t_min_) && (lte >= 1.0));
    }

    /**
     * @brief Velocity threshold for edge-triggered output, if configured.
     *
     * When set, output is written only on a rising crossing of this value (see
     * Writer::is_write_required) and all other adaptive output is suppressed.
     */
    std::optional<double> v_th() const { return v_th_; }

    /**
     * @brief Whether every step above v_th should be written, instead of only the crossing.
     *
     * Only meaningful when v_th() is set. When true, output is written on every monitor
     * step for which VMax >= v_th, rather than just once on the rising edge (see
     * Writer::is_write_required).
     */
    bool high_freq() const { return high_freq_; }

private:
    const double atol_;
    const double rtol_;
    const double t_min_;
    const double t_max_;
    const std::optional<double> v_th_;
    const bool high_freq_;
};

} // namespace tndm

#endif // ADAPTIVEOUTPUTSTRATEGY_20201119_H
