#ifndef TENSORPRODUCTRULE_20200615_H
#define TENSORPRODUCTRULE_20200615_H

#include "SimplexQuadratureRule.h"

#include <cstddef>

namespace tndm {

/**
 * @brief Returns quadrature rule on the reference simplex with n^D points and order 2n-1.
 *
 * The reference simplex is
 * D=1: [0,1]
 * D=2: \{x \in R^2 : 0 \leq x_1 \leq 1 \wedge 0 \leq x_2 \leq 1-x_1\}
 * D=3: \{x \in R^3 : 0 \leq x_1 \leq 1 \wedge 0 \leq x_2 \leq 1-x_1 \wedge 0 \leq x_3 \leq 1-x_2-x_3\}
 */
template <std::size_t D> SimplexQuadratureRule<D> tensorProductRule(unsigned n);

}

#endif // TENSORPRODUCTRULE_20200615_H
