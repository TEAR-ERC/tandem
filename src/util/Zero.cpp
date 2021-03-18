#include "Zero.h"

#include <cmath>
#include <cstdio>
#include <limits>

namespace tndm {

function_nan_inf::function_nan_inf(double x, double Fx) noexcept {
    snprintf(what_, MaxLen - 1, "NaN or Inf detected: F(x) = %lf with x = %lf", Fx, x);
}

double zeroIn(double a, double b, std::function<double(double)> F, double tol) {
    double eps = std::numeric_limits<double>::epsilon();
    double Fa = F(a);
    if (Fa == 0.0) {
        return a;
    }
    double Fb = F(b);

    if (std::isnan(Fa) || std::isinf(Fa)) {
        throw function_nan_inf(a, Fa);
    }
    if (std::isnan(Fb) || std::isinf(Fb)) {
        throw function_nan_inf(b, Fb);
    }
    if (Fb != 0.0 && std::copysign(Fa, Fb) == Fa) {
        throw std::logic_error("F(a) and F(b) must have different sign.");
    }
    double c = a;
    double Fc = Fa;
    double d = b - a;
    double e = d;
    while (Fb != 0.0) {
        if (std::copysign(Fb, Fc) == Fb) {
            c = a;
            Fc = Fa;
            d = b - a;
            e = d;
        }
        if (std::fabs(Fc) < std::fabs(Fb)) {
            a = b;
            b = c;
            c = a;
            Fa = Fb;
            Fb = Fc;
            Fc = Fa;
        }
        // Convergence test
        double xm = 0.5 * (c - b);
        double tol1 = 2.0 * eps * std::fabs(b) + 0.5 * tol;
        if (std::fabs(xm) <= tol1 || Fb == 0.0) {
            break;
        }
        if (std::fabs(e) < tol1 || std::fabs(Fa) <= std::fabs(Fb)) {
            // bisection
            d = xm;
            e = d;
        } else {
            double s = Fb / Fa;
            double p, q;
            if (a != c) {
                // linear interpolation
                q = Fa / Fc;
                double r = Fb / Fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            } else {
                // inverse quadratic interpolation
                p = 2.0 * xm * s;
                q = 1.0 - s;
            }
            if (p > 0) {
                q = -q;
            } else {
                p = -p;
            }
            if (2.0 * p < 3.0 * xm * q - std::fabs(tol1 * q) && p < std::fabs(0.5 * e * q)) {
                e = d;
                d = p / q;
            } else {
                // bisection
                d = xm;
                e = d;
            }
        }
        a = b;
        Fa = Fb;
        if (std::fabs(d) > tol1) {
            b += d;
        } else {
            b += std::copysign(tol1, xm);
        }
        Fb = F(b);
        if (std::isnan(Fb) || std::isinf(Fb)) {
            throw function_nan_inf(b, Fb);
        }
    }
    return b;
}

} // namespace tndm
