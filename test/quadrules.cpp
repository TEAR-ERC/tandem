#include "basis/Functions.h"
#include "quadrules/GaussJacobi.h"
#include "quadrules/TensorProductRule.h"
#include "util/Combinatorics.h"

#include "doctest.h"

#include <cmath>

using namespace tndm;

TEST_CASE("Quadrature") {
    SUBCASE("Gauss Jacobi vs Maple") { // Test from SeisSol
        auto rule = GaussJacobi(5, 1, 3);
        CHECK(rule.points()[0] == doctest::Approx(0.86698568210542769702));
        CHECK(rule.points()[1] == doctest::Approx(0.57652877512667440772));
        CHECK(rule.points()[2] == doctest::Approx(0.17976783188823737401));
        CHECK(rule.points()[3] == doctest::Approx(-.25499675973326581341));
        CHECK(rule.points()[4] == doctest::Approx(-.65399981510135937963));
        CHECK(rule.weights()[0] == doctest::Approx(0.18915446768616357329));
        CHECK(rule.weights()[1] == doctest::Approx(0.58714974961811369751));
        CHECK(rule.weights()[2] == doctest::Approx(0.57657004957734461768));
        CHECK(rule.weights()[3] == doctest::Approx(0.22255926867518051648));
        CHECK(rule.weights()[4] == doctest::Approx(0.024566464443197594119));
    }

    SUBCASE("Triangle vs Maple") { // Test from SeisSol
        auto rule = tensorProductRule<2u>(2);
        CHECK(rule.points()[0][0] == doctest::Approx(0.64494897427831780982));
        CHECK(rule.points()[1][0] == doctest::Approx(0.64494897427831780982));
        CHECK(rule.points()[2][0] == doctest::Approx(0.15505102572168219018));
        CHECK(rule.points()[3][0] == doctest::Approx(0.15505102572168219018));
        CHECK(rule.points()[0][1] == doctest::Approx(0.28001991549907407200));
        CHECK(rule.points()[1][1] == doctest::Approx(0.075031110222608118175));
        CHECK(rule.points()[2][1] == doctest::Approx(0.66639024601470138669));
        CHECK(rule.points()[3][1] == doctest::Approx(0.17855872826361642311));
        CHECK(rule.weights()[0] == doctest::Approx(0.090979309128011415315));
        CHECK(rule.weights()[1] == doctest::Approx(0.090979309128011415315));
        CHECK(rule.weights()[2] == doctest::Approx(0.15902069087198858472));
        CHECK(rule.weights()[3] == doctest::Approx(0.15902069087198858472));
    }

    auto testOrder = [](auto D) {
        for (unsigned n = 1; n < 12; ++n) {
            auto rule = tensorProductRule<D()>(n);
            auto& pts = rule.points();
            auto& wgts = rule.weights();
            unsigned maxDegree = 2u * n - 1u;
            unsigned dim = pts[0].size();
            // Test integrated all monomials x_1^{j_1} * ... * x_D^{j_D}
            for (auto&& j : AllIntegerSums<D()>(maxDegree)) {
                double integral = 0.0;
                for (std::size_t q = 0; q < pts.size(); ++q) {
                    double mono = 1.0;
                    for (std::size_t d = 0; d < D(); ++d) {
                        mono *= std::pow(pts[q][d], j[d]);
                    }
                    integral += mono * wgts[q];
                }
                // Solution is (at least up to D=3)
                // j_1!*...*j_D! / (j_1 + ... + j_D + D)! =
                //   j_1!*...*j_{D-1}! / ((j_{D-1} + 1)*...*(j_1 + ... + j_D + D))
                std::sort(j.begin(), j.end());
                auto jsum = std::accumulate(j.begin(), j.end(), 0u);
                double reference = 1.0 / rangeProduct(j[D()-1u] + 1u, jsum + D());
                for (std::size_t d = 0; d < D()-1u; ++d) {
                   reference *= factorial(j[d]);
                }
                CHECK(integral == doctest::Approx(reference));
            }
        }
    };

    SUBCASE("Interval quadrature order test") {
        testOrder(std::integral_constant<std::size_t, 1u>());
    }

    SUBCASE("Triangle quadrature order test") {
        testOrder(std::integral_constant<std::size_t, 2u>());
    }

    SUBCASE("Tetrahedron quadrature order test") {
        testOrder(std::integral_constant<std::size_t, 3u>());
    }
}
