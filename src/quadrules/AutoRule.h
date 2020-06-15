#ifndef AUTORULE_20200615_H
#define AUTORULE_20200615_H

#include "TensorProductRule.h"

namespace tndm {

/**
 * @brief Factory function for quadrature rules
 */
template <std::size_t D> auto simplexQuadratureRule(unsigned minQuadOrder) {
    return tndm::tensorProductRule<D>(1 + minQuadOrder / 2); // n = ceil((minQuadOrder+1)/2)
}

} // namespace tndm

#endif // AUTORULE_20200615_H
