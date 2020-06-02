// Modified from SeisSol project
/**
 * Copyright (c) 2017, SeisSol Group
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 **/
#ifndef QUADRATURE_H
#define QUADRATURE_H

#include <array>
#include <limits>
#include <vector>

namespace tndm {
namespace quadrature_settings {
static unsigned const MaxIterations = 100;
static double const Tolerance = 10. * std::numeric_limits<double>::epsilon();
} // namespace quadrature_settings

/**
 * @brief Contains a quadrature rule.
 *
 * int_{-1}^{1} f(y) dy \approx \sum_{i=0}^{n-1} f(points[i]) * weights[i]
 *
 */
class IntervalQuadratureRule {
public:
    IntervalQuadratureRule(unsigned n) : points_(n), weights_(n), a(-1.0), b(1.0) {}

    void changeInterval(double start, double stop);

    std::vector<double>& points() { return points_; }
    std::vector<double>& weights() { return weights_; }

    std::size_t size() const { return points_.size(); }

private:
    std::vector<double> points_;
    std::vector<double> weights_;
    double a, b;
};

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

/**
 * @brief Contains a quadrature rule for simplices.
 *
 */
template <std::size_t D> class SimplexQuadratureRule {
public:
    SimplexQuadratureRule(unsigned n) : points_(n), weights_(n) {}

    std::vector<std::array<double, D>>& points() { return points_; }
    std::vector<double>& weights() { return weights_; }

    std::size_t size() const { return points_.size(); }

private:
    std::vector<std::array<double, D>> points_;
    std::vector<double> weights_;
};

/**
 * @brief Returns quadrature rule on the reference triangle with vertices (0,0),(1,0),(0,1).
 */
SimplexQuadratureRule<2> TriangleQuadrature(unsigned n);

/**
 * @brief Returns quadrature rule on the reference tetrahedron with vertices
 * (0,0,0),(1,0,0),(0,1,0),(0,0,1).
 */
SimplexQuadratureRule<3> TetrahedronQuadrature(unsigned n);

} // namespace tndm

#endif // QUADRATURE_H
