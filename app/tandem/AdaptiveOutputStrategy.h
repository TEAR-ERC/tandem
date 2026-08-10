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
                           std::optional<double> v_th = std::nullopt,
                           std::optional<int> freq = std::nullopt)
        : atol_(atol), rtol_(rtol), t_min_(t_min), t_max_(t_max), v_th_(v_th), freq_(freq) {}

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
     * @brief Monitor-step period for periodic output above v_th, if configured.
     *
     * Only meaningful when v_th() is set. When set (must be a positive integer,
     * enforced at schema parse time), output is written every freq()-th monitor
     * step for as long as VMax >= v_th, rather than just once on the rising edge
     * (see Writer::is_write_required). This bounds how much output an excursion
     * above v_th can produce.
     *
     * When v_th() is set but this is left unset, the writer is in single-shot
     * mode: it writes exactly one snapshot on the first rising crossing of v_th
     * and the run is then stopped (see Writer::stop_after_write).
     */
    std::optional<int> freq() const { return freq_; }

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
