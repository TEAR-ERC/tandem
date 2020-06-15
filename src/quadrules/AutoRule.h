#ifndef AUTORULE_20200615_H
#define AUTORULE_20200615_H

#include "NME2020.h"
#include "TensorProductRule.h"

#include <cstddef>

namespace tndm {

/**
 * @brief Factory function for quadrature rules
 */
template <std::size_t D> auto simplexQuadratureRule(unsigned minQuadOrder) {
    return TensorProductRule<D>::get(minQuadOrder);
}

} // namespace tndm

#endif // AUTORULE_20200615_H
