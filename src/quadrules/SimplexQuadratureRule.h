#ifndef SIMPLEXQUADRATURERULE_20200615_H
#define SIMPLEXQUADRATURERULE_20200615_H

#include <array>
#include <vector>

namespace tndm {

/**
 * @brief Contains a quadrature rule for simplices.
 *
 */
template <std::size_t D> class SimplexQuadratureRule {
public:
    SimplexQuadratureRule() : order_(-1) {}
    SimplexQuadratureRule(unsigned n, int order) : points_(n), weights_(n), order_(order) {}

    std::vector<std::array<double, D>>& points() { return points_; }
    std::vector<double>& weights() { return weights_; }

    std::size_t size() const { return points_.size(); }
    int order() const { return order_; }

private:
    std::vector<std::array<double, D>> points_;
    std::vector<double> weights_;
    int order_;
};

} // namespace tndm

#endif // SIMPLEXQUADRATURERULE_20200615_H
