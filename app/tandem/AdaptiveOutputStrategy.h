#ifndef ADAPTIVEOUTPUTSTRATEGY_20201119_H
#define ADAPTIVEOUTPUTSTRATEGY_20201119_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>

namespace tndm {

/**
 * @brief Decides whether an output is due, based on the change of VMax and the elapsed time.
 *
 * Default rule: write if more than t_max has passed, or if more than t_min has passed and VMax
 * changed by about rtol (relative) / atol (absolute) with respect to its value at the last output.
 *
 * Optional threshold rule (v_th and freq both set): while VMax >= v_th the change-based criterion
 * is replaced by a fixed step interval: output at the first step of the excursion and then at
 * every time step whose number is a multiple of freq. As the time integrator takes smaller steps
 * where the solution changes quickly, the sampling follows the dynamics. This keeps sampling a
 * seismic event, where VMax stays roughly constant and the default rule would write almost
 * nothing. t_min and t_max still apply.
 *
 * The rule only depends on its arguments (time since last output, VMax at last output, VMax now,
 * time step number). The first three are stored in a checkpoint and the step number is restored
 * with the time solver, so no extra state is needed to restart.
 */
class AdaptiveOutputInterval {
public:
    AdaptiveOutputInterval(double atol, double rtol, double t_min, double t_max,
                           std::optional<double> v_th = std::nullopt,
                           std::optional<int> freq = std::nullopt)
        : atol_(atol), rtol_(rtol), t_min_(t_min), t_max_(t_max), v_th_(v_th), freq_(freq) {
        if (v_th_.has_value() != freq_.has_value()) {
            throw std::invalid_argument(
                "Output options v_th and freq must be given together (or both omitted).");
        }
    }

    bool operator()(double delta_time, double last_VMax, double VMax, std::size_t step) const {
        if (delta_time > t_max_) {
            return true;
        }
        if (delta_time <= t_min_) {
            return false;
        }
        if (v_th_ && VMax >= *v_th_) {
            // Above the threshold: first sample of the excursion (the last output was taken
            // below the threshold), then every freq-th time step.
            return last_VMax < *v_th_ || step % static_cast<std::size_t>(*freq_) == 0;
        }
        double tol = atol_ + std::max(last_VMax, VMax) * rtol_;
        double lte = std::abs(VMax - last_VMax) / tol;
        return lte >= 1.0;
    }

private:
    const double atol_;
    const double rtol_;
    const double t_min_;
    const double t_max_;
    const std::optional<double> v_th_;
    const std::optional<int> freq_;
};

} // namespace tndm

#endif // ADAPTIVEOUTPUTSTRATEGY_20201119_H
