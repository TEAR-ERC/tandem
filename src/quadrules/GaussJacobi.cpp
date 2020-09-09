#include "GaussJacobi.h"
#include "basis/Functions.h"
#include "quadrules/IntervalQuadratureRule.h"

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vector>

namespace tndm {

IntervalQuadratureRule GaussJacobi(unsigned n, unsigned a, unsigned b) {
    IntervalQuadratureRule rule(n);
    //                -(2n + a + b + 2) (n + a)! (n + b)! 2^(a+b)
    // weightFactor = -------------------------------------------
    //                   (n + a + b + 1) (n + a + b)! (n + 1)!
    double weightFactor = -(2.0 * n + a + b + 2) * rangeProduct(n + 1, n + b) * (1 << (a + b)) /
                          ((n + a + b + 1.0) * rangeProduct(n + a + 1, n + a + b) * (n + 1));
    for (unsigned i = 1; i <= n; ++i) {
        // x = Initial guess for polynomial root
        double x = cos(M_PI * (0.5 * a + i - 0.25) / (0.5 * (1.0 + a + b) + n));
        double Pn = 0.0, dPn = 1.0;
        unsigned it = 0;
        // Refine polynomial roots with Newton iteration
        do {
            x -= Pn / dPn;
            Pn = JacobiP(n, a, b, x);
            dPn = JacobiPDerivative(n, a, b, x);
        } while (fabs(Pn) > quadrature_settings::Tolerance &&
                 ++it < quadrature_settings::MaxIterations);
        rule.points()[i - 1] = x;
        rule.weights()[i - 1] = weightFactor / (JacobiP(n + 1, a, b, x) * dPn);
    }
    return rule;
}

} // namespace tndm
