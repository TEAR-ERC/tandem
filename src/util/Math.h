#ifndef MATH_20201105_H
#define MATH_20201105_H

#include <cstdint>

namespace tndm {

// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T> int sgn(T val) { return (T(0) < val) - (val < T(0)); }

/**
 * @brief Computes \prod_{i=from}^{to} i. Returns 1 if from > to.
 */
uint64_t rangeProduct(uint64_t from, uint64_t to);

/**
 * @brief Factorial operation
 *
 * @param n
 *
 * @return n!
 */
inline uint64_t factorial(uint64_t n) { return rangeProduct(1, n); }

} // namespace tndm

#endif // MATH_20201105_H
