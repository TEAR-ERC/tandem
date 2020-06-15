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
    SimplexQuadratureRule(unsigned n) : points_(n), weights_(n) {}

    std::vector<std::array<double, D>>& points() { return points_; }
    std::vector<double>& weights() { return weights_; }

    std::size_t size() const { return points_.size(); }

private:
    std::vector<std::array<double, D>> points_;
    std::vector<double> weights_;
};

}

#endif // SIMPLEXQUADRATURERULE_20200615_H
