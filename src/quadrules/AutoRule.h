#ifndef AUTORULE_20200615_H
#define AUTORULE_20200615_H

#include "TabulatedRules.h"
#include "TensorProductRule.h"

#include <cstddef>

namespace tndm {

template <typename Head> auto minSizeSimplexQuadratureRule(unsigned minQuadOrder) {
    return Head::get(minQuadOrder);
}

template <typename C1, typename C2, typename... Tail>
auto minSizeSimplexQuadratureRule(unsigned minQuadOrder) {
    if (C1::size(minQuadOrder) <= C2::size(minQuadOrder)) {
        return minSizeSimplexQuadratureRule<C1, Tail...>(minQuadOrder);
    }
    return minSizeSimplexQuadratureRule<C2, Tail...>(minQuadOrder);
}

/**
 * @brief Automatically select quadrature rule on reference simplex.
 *
 * The reference simplex is
 *
 * \f$D=1: [0,1]\f$
 *
 * \f$D=2: \{x \in R^2 : 0 \leq x_1 \leq 1 \wedge 0 \leq x_1+x_2 \leq 1\}\f$
 *
 * \f$D=3: \{x \in R^3 : 0 \leq x_1 \leq 1 \wedge 0 \leq x_1+x_2 \leq 1 \wedge 0 \leq x_1+x_2+x_3
 * \leq 1\}\f$
 *
 * @tparam D Dimension
 * @param minQuadOrder All polynomials with maximum degree minQuadOrder are integrated exactly up to
 * round-off errors.
 *
 * @return SimplexQuadratureRule
 */
template <std::size_t D> auto simplexQuadratureRule(unsigned minQuadOrder) {
    if constexpr (D == 2u) {
        return minSizeSimplexQuadratureRule<WitherdenVincent2015<2u>, TensorProductRule<2u>>(
            minQuadOrder);
    } else if constexpr (D == 3u) {
        return minSizeSimplexQuadratureRule<JaskowiecSukumar2020, WitherdenVincent2015<3u>,
                                            TensorProductRule<3u>>(minQuadOrder);
    }
    return TensorProductRule<D>::get(minQuadOrder);
}

} // namespace tndm

#endif // AUTORULE_20200615_H
