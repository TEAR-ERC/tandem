#ifndef TENSORPRODUCTRULE_20200615_H
#define TENSORPRODUCTRULE_20200615_H

#include "SimplexQuadratureRule.h"

#include <cstddef>

namespace tndm {

template <std::size_t D> class TensorProductRule {
public:
    static unsigned pointsPerDim(unsigned minQuadOrder) {
        return (1 + minQuadOrder / 2); // n = ceil((minQuadOrder+1)/2)
    }

    /**
     * @brief Returns size of rule with at least minQuadOrder.
     */
    static std::size_t size(unsigned minQuadOrder) {
        auto n = pointsPerDim(minQuadOrder);
        std::size_t s = 1u;
        for (std::size_t d = 0; d < D; ++d) {
            s *= n;
        }
        return s;
    }

    /**
     * @brief Returns quadrature rule on the reference simplex with n^D points and order 2n-1.
     *
     * The reference simplex is
     * D=1: [0,1]
     * D=2: \{x \in R^2 : 0 \leq x_1 \leq 1 \wedge 0 \leq x_1+x_2 \leq 1\}
     * D=3: \{x \in R^3 : 0 \leq x_1 \leq 1 \wedge 0 \leq x_1+x_2 \leq 1 \wedge 0 \leq x_1+x_2+x_3
     * \leq 1\}
     */
    static SimplexQuadratureRule<D> get(unsigned minQuadOrder);
};

} // namespace tndm

#endif // TENSORPRODUCTRULE_20200615_H
