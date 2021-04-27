#ifndef MATH_20201105_H
#define MATH_20201105_H

namespace tndm {

// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T> int sgn(T val) { return (T(0) < val) - (val < T(0)); }

} // namespace tndm

#endif // MATH_20201105_H
