#ifndef INTERVALQUADRATURERULE_20200615_H
#define INTERVALQUADRATURERULE_20200615_H

#include <cstddef>
#include <vector>

namespace tndm {

/**
 * @brief Contains a quadrature rule.
 *
 * int_{-1}^{1} f(y) dy \approx \sum_{i=0}^{n-1} f(points[i]) * weights[i]
 *
 */
class IntervalQuadratureRule {
public:
    IntervalQuadratureRule(unsigned n) : points_(n), weights_(n), a(-1.0), b(1.0) {}

    void changeInterval(double start, double stop);

    std::vector<double>& points() { return points_; }
    std::vector<double>& weights() { return weights_; }

    std::size_t size() const { return points_.size(); }

private:
    std::vector<double> points_;
    std::vector<double> weights_;
    double a, b;
};

} // namespace tndm

#endif // INTERVALQUADRATURERULE_20200615_H
