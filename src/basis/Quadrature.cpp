#include "Quadrature.h"
#include "Functions.h"

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tndm {

void IntervalQuadratureRule::changeInterval(double start, double stop) {
    double diff = 0.5 * (b - a);
    double avg = 0.5 * (a + b);
    a = start;
    b = stop;
    double newDiff = 0.5 * (b - a);
    double newAvg = 0.5 * (a + b);
    for (auto& p : points_) {
        p = newDiff * (p - avg) / diff + newAvg;
    }
    for (auto& w : weights_) {
        w *= newDiff / diff;
    }
}

IntervalQuadratureRule GaussJacobi(unsigned n, unsigned a, unsigned b) {
    IntervalQuadratureRule rule(n);
    double weightFactor = -(2.0 * n + a + b + 2) * factorial(n + a) * factorial(n + b) *
                          (1 << (a + b)) /
                          ((n + a + b + 1.0) * factorial(n + a + b) * factorial(n + 1));
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

SimplexQuadratureRule<1> IntervalQuadrature(unsigned n) {
    SimplexQuadratureRule<1> rule(n);

    auto gj = GaussJacobi(n, 0, 0);
    gj.changeInterval(0.0, 1.0);

    for (std::size_t i = 0; i < n; ++i) {
        rule.points()[i][0] = gj.points()[i];
        rule.weights()[i] = gj.weights()[i];
    }

    return rule;
}

SimplexQuadratureRule<2> TriangleQuadrature(unsigned n) {
    SimplexQuadratureRule<2> rule(n * n);

    auto gj0 = GaussJacobi(n, 0, 0);
    auto gj1 = GaussJacobi(n, 1, 0);

    gj0.changeInterval(0.0, 1.0);
    gj1.changeInterval(0.0, 1.0);

    std::size_t idx = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            rule.points()[idx][0] = gj1.points()[i];
            rule.points()[idx][1] = gj0.points()[j] * (1.0 - gj1.points()[i]);
            rule.weights()[idx] = 0.5 * gj1.weights()[i] * gj0.weights()[j];
            ++idx;
        }
    }

    return rule;
}

SimplexQuadratureRule<3> TetrahedronQuadrature(unsigned n) {
    SimplexQuadratureRule<3> rule(n * n * n);

    auto gj0 = GaussJacobi(n, 0, 0);
    auto gj1 = GaussJacobi(n, 1, 0);
    auto gj2 = GaussJacobi(n, 2, 0);

    gj0.changeInterval(0.0, 1.0);
    gj1.changeInterval(0.0, 1.0);
    gj2.changeInterval(0.0, 1.0);

    std::size_t idx = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                rule.points()[idx][0] = gj2.points()[i];
                rule.points()[idx][1] = gj1.points()[j] * (1 - gj2.points()[i]);
                rule.points()[idx][2] =
                    gj0.points()[k] * (1 - gj1.points()[j]) * (1 - gj2.points()[i]);
                rule.weights()[idx] =
                    0.125 * gj2.weights()[i] * gj1.weights()[j] * gj0.weights()[k];
                ++idx;
            }
        }
    }

    return rule;
}

} // namespace tndm
