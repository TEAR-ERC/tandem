#ifndef ZERO_20200826_H
#define ZERO_20200826_H

#include <functional>
#include <stdexcept>

namespace tndm {

class function_nan_inf : public std::exception {
public:
    function_nan_inf(double x, double Fx) noexcept;
    virtual ~function_nan_inf() {}
    virtual const char* what() const noexcept { return what_; }

private:
    static constexpr std::size_t MaxLen = 128;
    char what_[MaxLen];
};

/**
 * @brief Find zero of function F in the interval [a,b].
 *
 * Port of Fortran routine ZEROIN from
 * http://www.netlib.org/fmm/zeroin.f
 *
 * See also G. Forsythe, M. Malcolm, C. Moler, "Computer Methods for Mathematical Computations",
 * Prentice-Hall, 1977.
 *
 * F(a) and F(b) must have opposite signs. The returned result lies in the interval [b,c] with
 * |b - c| <= tol + 4.0 * epsilon * |b|
 * (Passing tol = 0 is fine.)
 *
 * @param a Interval start
 * @param b Interval end
 * @param F Function of x
 * @param tol Absolute tolerance
 *
 * @return x with f(x) = 0
 */
double zeroIn(double a, double b, std::function<double(double)> F, double tol = 0.0);

} // namespace tndm

#endif // ZERO_20200826_H
