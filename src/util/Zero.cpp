#include "Zero.h"

#include <cassert>
#include <cmath>
#include <limits>

namespace tndm {

double zeroIn(double a, double b, std::function<double(double)> F, double tol) {
    double eps = std::numeric_limits<double>::epsilon();
    double Fa = F(a);
    double Fb = F(b);
    assert(!std::isnan(Fa));
    assert(!std::isnan(Fb));
    assert(std::copysign(Fa, Fb) != Fa); // Fa and Fb have different signs
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
        assert(!std::isnan(Fb));
    }
    return b;
}

} // namespace tndm
