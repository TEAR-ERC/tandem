#include "basis/Functions.h"
#include "doctest.h"

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

using tndm::gradTetraDubinerP;
using tndm::TetraDubinerP;

using TetraBasisFunction = std::function<double(std::array<double, 3> const&)>;
using GradTetraBasisFunction = std::function<std::array<double, 3>(std::array<double, 3> const&)>;

TEST_CASE("Basis") {
    std::vector<std::array<double, 3>> testPoints{
        {0.0, 0.0, 0.0},    {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
        {0.25, 0.25, 0.25}, {0.1, 0.1, 0.1}, {0.1, 0.2, 0.3}, {0.2, 0.1, 0.3}};

    std::vector<std::pair<std::array<unsigned, 3>, TetraBasisFunction>> tetraBFs{
        {{0, 0, 0}, [](std::array<double, 3> const& p) { return 1.0; }},
        {{1, 0, 0}, [](std::array<double, 3> const& p) { return -1.0 + 2.0 * p[0] + p[1] + p[2]; }},
        {{0, 1, 0}, [](std::array<double, 3> const& p) { return -1.0 + 3.0 * p[1] + p[2]; }},
        {{0, 0, 1}, [](std::array<double, 3> const& p) { return -1.0 + 4.0 * p[2]; }},
        {{2, 0, 0},
         [](std::array<double, 3> const& p) {
             return 1.0 - 6.0 * p[0] + 6.0 * p[0] * p[0] - 2.0 * p[1] + 6.0 * p[0] * p[1] +
                    p[1] * p[1] - 2.0 * p[2] + 6.0 * p[0] * p[2] + 2.0 * p[1] * p[2] + p[2] * p[2];
         }},
        {{1, 1, 0},
         [](std::array<double, 3> const& p) {
             return 1.0 - 2.0 * p[0] - 6.0 * p[1] + 10.0 * p[0] * p[1] + 5.0 * p[1] * p[1] -
                    2.0 * p[2] + 2.0 * p[0] * p[2] + 6.0 * p[1] * p[2] + p[2] * p[2];
         }},
        {{0, 2, 0},
         [](std::array<double, 3> const& p) {
             return 1.0 - 8.0 * p[1] + 10.0 * p[1] * p[1] - 2.0 * p[2] + 8.0 * p[1] * p[2] +
                    p[2] * p[2];
         }},
        {{1, 0, 1},
         [](std::array<double, 3> const& p) {
             return 1.0 - 2.0 * p[0] - p[1] - 7.0 * p[2] + 12.0 * p[0] * p[2] + 6.0 * p[1] * p[2] +
                    6.0 * p[2] * p[2];
         }},
        {{0, 1, 1},
         [](std::array<double, 3> const& p) {
             return 1.0 - 3.0 * p[1] - 7.0 * p[2] + 18.0 * p[1] * p[2] + 6.0 * p[2] * p[2];
         }},
        {{0, 0, 2},
         [](std::array<double, 3> const& p) { return 1.0 - 10.0 * p[2] + 15.0 * p[2] * p[2]; }},
    };

    SUBCASE("Dubiner polynomials on tetrahedron") {
        for (auto& t : tetraBFs) {
            for (auto& p : testPoints) {
                CHECK(TetraDubinerP(t.first[0], t.first[1], t.first[2], p[0], p[1], p[2]) ==
                      doctest::Approx(t.second(p)));
            }
        }
    }

    std::vector<std::pair<std::array<unsigned, 3>, GradTetraBasisFunction>> gradTetraBFs{
        {{0, 0, 0},
         [](std::array<double, 3> const& p) -> std::array<double, 3> {
             return {0.0, 0.0, 0.0};
         }},
        {{1, 0, 0},
         [](std::array<double, 3> const& p) -> std::array<double, 3> {
             return {2.0, 1.0, 1.0};
         }},
        {{0, 1, 0},
         [](std::array<double, 3> const& p) -> std::array<double, 3> {
             return {0.0, 3.0, 1.0};
         }},
        {{0, 0, 1},
         [](std::array<double, 3> const& p) -> std::array<double, 3> {
             return {0.0, 0.0, 4.0};
         }},
        {{2, 0, 0},
         [](std::array<double, 3> const& p) -> std::array<double, 3> {
             return {-6.0 + 12.0 * p[0] + 6.0 * p[1] + 6.0 * p[2],
                     -2.0 + 6.0 * p[0] + 2.0 * p[1] + 2.0 * p[2],
                     -2.0 + 6.0 * p[0] + 2.0 * p[1] + 2.0 * p[2]};
         }},
        {{1, 1, 0},
         [](std::array<double, 3> const& p) -> std::array<double, 3> {
             return {-2.0 + 10.0 * p[1] + 2.0 * p[2], -6.0 + 10.0 * p[0] + 10.0 * p[1] + 6.0 * p[2],
                     -2.0 + 2.0 * p[0] + 6.0 * p[1] + 2.0 * p[2]};
         }},
        {{0, 2, 0},
         [](std::array<double, 3> const& p) -> std::array<double, 3> {
             return {0.0, -8.0 + 20.0 * p[1] + 8.0 * p[2], -2.0 + 8.0 * p[1] + 2.0 * p[2]};
         }},
        {{1, 0, 1},
         [](std::array<double, 3> const& p) -> std::array<double, 3> {
             return {-2.0 + 12.0 * p[2], -1.0 + 6.0 * p[2],
                     -7.0 + 12.0 * p[0] + 6.0 * p[1] + 12.0 * p[2]};
         }},
        {{0, 1, 1},
         [](std::array<double, 3> const& p) -> std::array<double, 3> {
             return {0.0, -3.0 + 18.0 * p[2], -7.0 + 18.0 * p[1] + 12.0 * p[2]};
         }},
        {{0, 0, 2},
         [](std::array<double, 3> const& p) -> std::array<double, 3> {
             return {0.0, 0.0, -10.0 + 30.0 * p[2]};
         }},

    };

    SUBCASE("Dubiner polynomials gradients on tetrahedron") {
        for (auto& t : gradTetraBFs) {
            for (auto& p : testPoints) {
                auto grad = gradTetraDubinerP(t.first[0], t.first[1], t.first[2], p[0], p[1], p[2]);
                auto refGrad = t.second(p);
                for (std::size_t d = 0; d < grad.size(); ++d) {
                    CHECK(grad[d] == doctest::Approx(refGrad[d]));
                }
            }
        }
    }
}
