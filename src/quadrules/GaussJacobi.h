#ifndef GAUSSJACOBI_20200615_H
#define GAUSSJACOBI_20200615_H

#include "IntervalQuadratureRule.h"

#include <limits>

namespace tndm {

namespace quadrature_settings {
static unsigned const MaxIterations = 100;
static double const Tolerance = 10. * std::numeric_limits<double>::epsilon();
} // namespace quadrature_settings

/** Returns quadrature points for the interval [-1,1] with weight function (1-x)^a * (1+x)^b, i.e.
 *  int_{-1}^{1} f(y)dy = sum_{i=0}^{n-1} f(points[i]) * weights[i]
 */
/**
 * @brief Returns Gauss Jacobi quadrature rule.
 *
 * The rule has n points on the interval [-1,1] and has weight function (1-x)^a (1+x)^b.
 *
 */
IntervalQuadratureRule GaussJacobi(unsigned n, unsigned a, unsigned b);

}

#endif // GAUSSJACOBI_20200615_H
