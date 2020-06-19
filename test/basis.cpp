#include "basis/Functions.h"
#include "basis/Nodal.h"
#include "basis/WarpAndBlend.h"

#include "doctest.h"

#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

using namespace tndm;

using TriBasisFunction = std::function<double(std::array<double, 2> const&)>;
using GradTriBasisFunction = std::function<std::array<double, 2>(std::array<double, 2> const&)>;
using TetraBasisFunction = std::function<double(std::array<double, 3> const&)>;
using GradTetraBasisFunction = std::function<std::array<double, 3>(std::array<double, 3> const&)>;

TEST_CASE("Basis") {
    const std::vector<double> testPoints1{0.0, 1.0, 0.25, 0.1, 0.2};
    const double divisor = 2.0;

    SUBCASE("Test singularity free JacobiP against JacobiP") {
        for (unsigned n = 0; n < 5; ++n) {
            for (unsigned a = 0; a < 10; ++a) {
                for (unsigned b = 0; b < 4; ++b) {
                    for (auto& point : testPoints1) {
                        CHECK(JacobiP(n, a, b, point / divisor) * std::pow(divisor, n) ==
                              SingularityFreeJacobiP(n, a, b, point, divisor));
                    }
                }
            }
        }
    }

    std::vector<std::array<double, 2>> testPoints2{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {0.25, 0.25},
                                                   {0.1, 0.1}, {0.1, 0.2}, {0.2, 0.1}};

    std::vector<std::pair<std::array<unsigned, 2>, TriBasisFunction>> triBFs{
        {{0, 0}, [](std::array<double, 2> const& p) { return 1.0; }},
        {{1, 0}, [](std::array<double, 2> const& p) { return -1.0 + 2.0 * p[0] + p[1]; }},
        {{0, 1}, [](std::array<double, 2> const& p) { return -1.0 + 3.0 * p[1]; }},
        {{2, 0},
         [](std::array<double, 2> const& p) {
             return 1.0 - 6.0 * p[0] + 6.0 * p[0] * p[0] - 2.0 * p[1] + 6.0 * p[0] * p[1] +
                    p[1] * p[1];
         }},
        {{1, 1},
         [](std::array<double, 2> const& p) {
             return 1.0 - 2.0 * p[0] - 6.0 * p[1] + 10.0 * p[0] * p[1] + 5.0 * p[1] * p[1];
         }},
        {{0, 2},
         [](std::array<double, 2> const& p) { return 1.0 - 8.0 * p[1] + 10.0 * p[1] * p[1]; }},
        {{3, 0},
         [](std::array<double, 2> const& p) {
             return -1.0 + 12.0 * p[0] - 30.0 * p[0] * p[0] + 20.0 * p[0] * p[0] * p[0] +
                    3.0 * p[1] - 24.0 * p[0] * p[1] + 30.0 * p[0] * p[0] * p[1] -
                    3.0 * p[1] * p[1] + 12.0 * p[0] * p[1] * p[1] + p[1] * p[1] * p[1];
         }},
        {{0, 3}, [](std::array<double, 2> const& p) {
             return -1.0 + 15.0 * p[1] - 45.0 * p[1] * p[1] + 35.0 * p[1] * p[1] * p[1];
         }}};

    SUBCASE("Dubiner polynomials on triangle") {
        for (auto& t : triBFs) {
            for (auto& p : testPoints2) {
                CHECK(TriDubinerP(t.first, p) == doctest::Approx(t.second(p)));
            }
        }
    }

    std::vector<std::pair<std::array<unsigned, 2>, GradTriBasisFunction>> gradTriBFs{
        {{0, 0},
         [](std::array<double, 2> const& p) -> std::array<double, 2> {
             return {0.0, 0.0};
         }},
        {{1, 0},
         [](std::array<double, 2> const& p) -> std::array<double, 2> {
             return {2.0, 1.0};
         }},
        {{0, 1},
         [](std::array<double, 2> const& p) -> std::array<double, 2> {
             return {0.0, 3.0};
         }},
        {{2, 0},
         [](std::array<double, 2> const& p) -> std::array<double, 2> {
             return {-6.0 + 12.0 * p[0] + 6.0 * p[1], -2.0 + 6.0 * p[0] + 2.0 * p[1]};
         }},
        {{1, 1},
         [](std::array<double, 2> const& p) -> std::array<double, 2> {
             return {-2.0 + 10.0 * p[1], -6.0 + 10.0 * p[0] + 10.0 * p[1]};
         }},
        {{0, 2},
         [](std::array<double, 2> const& p) -> std::array<double, 2> {
             return {0.0, -8.0 + 20.0 * p[1]};
         }},
        {{0, 3}, [](std::array<double, 2> const& p) -> std::array<double, 2> {
             return {0.0, 15.0 - 90.0 * p[1] + 105.0 * p[1] * p[1]};
         }}};

    SUBCASE("Dubiner polynomials gradients on triangle") {
        for (auto& t : gradTriBFs) {
            for (auto& p : testPoints2) {
                auto grad = gradTriDubinerP(t.first, p);
                auto refGrad = t.second(p);
                for (std::size_t d = 0; d < grad.size(); ++d) {
                    CHECK(grad[d] == doctest::Approx(refGrad[d]));
                }
            }
        }
    }

    std::vector<std::array<double, 3>> testPoints3{
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
         [](std::array<double, 3> const& p) { return 1.0 - 10.0 * p[2] + 15.0 * p[2] * p[2]; }}};

    SUBCASE("Dubiner polynomials on tetrahedron") {
        for (auto& t : tetraBFs) {
            for (auto& p : testPoints3) {
                CHECK(TetraDubinerP(t.first, p) == doctest::Approx(t.second(p)));
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
        {{0, 0, 2}, [](std::array<double, 3> const& p) -> std::array<double, 3> {
             return {0.0, 0.0, -10.0 + 30.0 * p[2]};
         }}};

    SUBCASE("Dubiner polynomials gradients on tetrahedron") {
        for (auto& t : gradTetraBFs) {
            for (auto& p : testPoints3) {
                auto grad = gradTetraDubinerP(t.first, p);
                auto refGrad = t.second(p);
                for (std::size_t d = 0; d < grad.size(); ++d) {
                    CHECK(grad[d] == doctest::Approx(refGrad[d]));
                }
            }
        }
    }
}

TEST_CASE("Nodes") {
    SUBCASE("LGL nodes") {
        auto glPoints2 = LegendreGaussLobattoPoints(2, 0, 0);
        CHECK(glPoints2.size() == 2);
        CHECK(glPoints2[0] == doctest::Approx(-1.0));
        CHECK(glPoints2[1] == doctest::Approx(1.0));

        auto glPoints3 = LegendreGaussLobattoPoints(3, 0, 0);
        CHECK(glPoints3.size() == 3);
        CHECK(glPoints3[0] == doctest::Approx(-1.0));
        CHECK(glPoints3[1] == doctest::Approx(0.0));
        CHECK(glPoints3[2] == doctest::Approx(1.0));

        auto glPoints5 = LegendreGaussLobattoPoints(5, 0, 0);
        CHECK(glPoints5.size() == 5);
        CHECK(glPoints5[0] == doctest::Approx(-1.0));
        CHECK(glPoints5[1] == doctest::Approx(-sqrt(3.0 / 7.0)));
        CHECK(glPoints5[2] == doctest::Approx(0.0));
        CHECK(glPoints5[3] == doctest::Approx(sqrt(3.0 / 7.0)));
        CHECK(glPoints5[4] == doctest::Approx(1.0));
    }

    auto checkNodes = [](auto const& nodes, auto const& refNodes) {
        REQUIRE(nodes.size() == refNodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            for (std::size_t j = 0; j < nodes[i].size(); ++j) {
                CHECK(nodes[i][j] == doctest::Approx(refNodes[i][j]));
            }
        }
    };

    auto wab2 = WarpAndBlendFactory<2>();
    auto wab3 = WarpAndBlendFactory<3>();

    SUBCASE("Triangle nodes P1") {
        std::vector<std::array<double, 2>> refNodes{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
        checkNodes(wab2(1), refNodes);
    }

    SUBCASE("Triangle nodes P2") {
        std::vector<std::array<double, 2>> refNodes{{0.0, 0.0}, {0.5, 0.0}, {0.0, 0.5},
                                                    {1.0, 0.0}, {0.5, 0.5}, {0.0, 1.0}};
        checkNodes(wab2(2), refNodes);
    }

    SUBCASE("Triangle nodes P3") {
        std::vector<std::array<double, 2>> refNodes{
            {0.00000000000000000000, 0.00000000000000000000},
            {0.27639320225002095288, 0.00000000000000000000},
            {0.00000000000000000000, 0.27639320225002095288},
            {0.72360679774997904712, 0.00000000000000000000},
            {0.33333333333333325932, 0.33333333333333325932},
            {0.00000000000000000000, 0.72360679774997882507},
            {1.00000000000000000000, 0.00000000000000000000},
            {0.72360679774997893610, 0.27639320225002106390},
            {0.27639320225002095288, 0.72360679774997904712},
            {0.00000000000000000000, 1.00000000000000000000}};
        checkNodes(wab2(3), refNodes);
    }

    SUBCASE("Triangle nodes P4") {
        std::vector<std::array<double, 2>> refNodes{
            {0.00000000000000000000, 0.00000000000000000000},
            {0.17267316464601145665, 0.00000000000000000000},
            {0.00000000000000000000, 0.17267316464601145665},
            {0.50000000000000000000, 0.00000000000000000000},
            {0.22420824622234719614, 0.22420824622234719614},
            {0.00000000000000000000, 0.50000000000000000000},
            {0.82732683535398865438, 0.00000000000000005551},
            {0.55158350755530560772, 0.22420824622234719614},
            {0.22420824622234719614, 0.55158350755530560772},
            {0.00000000000000005551, 0.82732683535398854335},
            {1.00000000000000000000, 0.00000000000000000000},
            {0.82732683535398865438, 0.17267316464601140114},
            {0.50000000000000000000, 0.50000000000000000000},
            {0.17267316464601140114, 0.82732683535398865438},
            {0.00000000000000000000, 1.00000000000000000000}};
        checkNodes(wab2(4), refNodes);
    }

    SUBCASE("Tet nodes P1") {
        std::vector<std::array<double, 3>> refNodes{
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        checkNodes(wab3(1), refNodes);
    }

    SUBCASE("Tet nodes P2") {
        std::vector<std::array<double, 3>> refNodes{
            {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.5}, {1.0, 0.0, 0.0},
            {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 1.0, 0.0}, {0.0, 0.5, 0.5}, {0.0, 0.0, 1.0}};
        checkNodes(wab3(2), refNodes);
    }

    SUBCASE("Tet nodes P3") {
        std::vector<std::array<double, 3>> refNodes{
            {0.00000000000000000000, 0.00000000000000000000, 0.00000000000000005551},
            {0.27639320225002095288, 0.00000000000000000000, 0.00000000000000005551},
            {-0.00000000000000011102, 0.27639320225002100839, 0.00000000000000005551},
            {-0.00000000000000011102, 0.00000000000000000000, 0.27639320225002095288},
            {0.72360679774997893610, 0.00000000000000000000, 0.00000000000000005551},
            {0.33333333333333337034, 0.33333333333333331483, 0.00000000000000005551},
            {0.33333333333333325932, 0.00000000000000005551, 0.33333333333333337034},
            {0.00000000000000000000, 0.72360679774997893610, 0.00000000000000005551},
            {0.00000000000000000000, 0.33333333333333331483, 0.33333333333333337034},
            {0.00000000000000000000, 0.00000000000000005551, 0.72360679774997882507},
            {1.00000000000000000000, 0.00000000000000000000, 0.00000000000000005551},
            {0.72360679774997882507, 0.27639320225002100839, 0.00000000000000005551},
            {0.72360679774997893610, 0.00000000000000000000, 0.27639320225002100839},
            {0.27639320225002100839, 0.72360679774997893610, 0.00000000000000005551},
            {0.33333333333333325932, 0.33333333333333331483, 0.33333333333333337034},
            {0.27639320225002095288, 0.00000000000000005551, 0.72360679774997893610},
            {0.00000000000000000000, 1.00000000000000000000, 0.00000000000000005551},
            {0.00000000000000000000, 0.72360679774997893610, 0.27639320225002106390},
            {0.00000000000000000000, 0.27639320225002100839, 0.72360679774997893610},
            {0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000}};
        checkNodes(wab3(3), refNodes);
    }

    SUBCASE("Tet nodes P4") {
        std::vector<std::array<double, 3>> refNodes{
            {0.00000000000000000000, 0.00000000000000000000, 0.00000000000000005551},
            {0.17267316464601134562, 0.00000000000000000000, 0.00000000000000005551},
            {0.00000000000000000000, 0.17267316464601129011, 0.00000000000000005551},
            {0.00000000000000000000, 0.00000000000000000000, 0.17267316464601145665},
            {0.50000000000000000000, 0.00000000000000000000, 0.00000000000000005551},
            {0.22420821395450318825, 0.22420821395450318825, 0.00000000000000005551},
            {0.22420821395450318825, 0.00000000000000000000, 0.22420821395450329927},
            {0.00000000000000000000, 0.50000000000000000000, 0.00000000000000005551},
            {-0.00000000000000011102, 0.22420821395450324376, 0.22420821395450329927},
            {0.00000000000000000000, 0.00000000000000000000, 0.50000000000000000000},
            {0.82732683535398854335, 0.00000000000000000000, 0.00000000000000005551},
            {0.55158357209099351248, 0.22420821395450318825, 0.00000000000000005551},
            {0.55158357209099351248, 0.00000000000000000000, 0.22420821395450324376},
            {0.22420821395450324376, 0.55158357209099351248, 0.00000000000000005551},
            {0.25000000000000000000, 0.25000000000000000000, 0.25000000000000000000},
            {0.22420821395450318825, 0.00000000000000000000, 0.55158357209099351248},
            {0.00000000000000000000, 0.82732683535398865438, 0.00000000000000005551},
            {0.00000000000000005551, 0.55158357209099351248, 0.22420821395450324376},
            {0.00000000000000000000, 0.22420821395450329927, 0.55158357209099351248},
            {0.00000000000000000000, 0.00000000000000005551, 0.82732683535398865438},
            {1.00000000000000000000, 0.00000000000000000000, 0.00000000000000005551},
            {0.82732683535398865438, 0.17267316464601129011, 0.00000000000000005551},
            {0.82732683535398865438, 0.00000000000000000000, 0.17267316464601145665},
            {0.50000000000000000000, 0.50000000000000000000, 0.00000000000000005551},
            {0.55158357209099351248, 0.22420821395450324376, 0.22420821395450329927},
            {0.50000000000000000000, 0.00000000000000000000, 0.50000000000000000000},
            {0.17267316464601134562, 0.82732683535398865438, 0.00000000000000005551},
            {0.22420821395450324376, 0.55158357209099351248, 0.22420821395450324376},
            {0.22420821395450318825, 0.22420821395450329927, 0.55158357209099351248},
            {0.17267316464601134562, 0.00000000000000005551, 0.82732683535398865438},
            {0.00000000000000000000, 1.00000000000000000000, 0.00000000000000005551},
            {0.00000000000000000000, 0.82732683535398865438, 0.17267316464601140114},
            {0.00000000000000000000, 0.50000000000000000000, 0.50000000000000000000},
            {0.00000000000000000000, 0.17267316464601140114, 0.82732683535398865438},
            {0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000},
        };
        checkNodes(wab3(4), refNodes);
    }
}
