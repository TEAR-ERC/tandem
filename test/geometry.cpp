#include "form/BC.h"
#include "geometry/Curvilinear.h"
#include "geometry/PointLocator.h"
#include "geometry/SimplexDistance.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/Simplex.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"

#include "doctest.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#define _USE_MATH_DEFINES
#include <cmath>

using namespace tndm;

TEST_CASE("Curvilinear") {
    double c = cos(3.14159265359 / 4.0);
    double s = sin(3.14159265359 / 4.0);

    constexpr std::size_t D = 2;
    std::array<uint64_t, 2> N = {1, 1};
    // scale to [-1,1] and rotate clockwise
    auto transform = [c, s](GenMesh<D>::vertex_t const& v) {
        double x = 2.0 * v[0] - 1.0;
        double y = 2.0 * v[1] - 1.0;
        return GenMesh<D>::vertex_t{c * x - s * y, c * x + s * y};
    };

    std::array<std::pair<BC, BC>, D> BCs;
    BCs.fill(std::make_pair(BC::None, BC::None));
    GenMesh<D> meshGen(N, BCs);
    auto globalMesh = meshGen.uniformMesh();
    auto mesh = globalMesh->getLocalMesh();
    Curvilinear<D> cl(*mesh, transform);

    SUBCASE("Map") {
        std::vector<std::tuple<std::size_t, std::array<double, 2>, std::array<double, 2>>> test{
            {0, {0.0, 0.0}, {-1.0, -1.0}}, {0, {1.0, 0.0}, {1.0, -1.0}},
            {0, {0.0, 1.0}, {-1.0, 1.0}},  {1, {0.0, 0.0}, {1.0, -1.0}},
            {1, {1.0, 0.0}, {-1.0, 1.0}},  {1, {0.0, 1.0}, {1.0, 1.0}}};
        for (auto& [elNo, xi, x] : test) {
            auto E = cl.evaluateBasisAt({xi});
            auto result = Managed(cl.mapResultInfo(1u));
            cl.map(elNo, E, result);
            CHECK(result(0, 0) == doctest::Approx(c * x[0] - s * x[1]));
            CHECK(result(1, 0) == doctest::Approx(c * x[0] + s * x[1]));
        }
    }

    SUBCASE("det(J)") {
        auto gradE = cl.evaluateGradientAt({{0.25, 0.25}});
        auto J = Managed(cl.jacobianResultInfo(1u));
        auto detJ = Managed(cl.detJResultInfo(1u));
        cl.jacobian(0, gradE, J);
        cl.detJ(0, J, detJ);
        CHECK(std::fabs(detJ(0)) == doctest::Approx(4.0));

        cl.jacobian(1, gradE, J);
        cl.detJ(1, J, detJ);
        CHECK(std::fabs(detJ(0)) == doctest::Approx(4.0));
    }

    SUBCASE("Normal") {
        using test_t =
            std::tuple<std::size_t, std::size_t, std::array<double, 2>, std::array<double, 2>>;
        /*
         *       +
         *     2   1
         *   /   1   \
         *  -----0-----
         *  -----2-----
         *   \   0   /
         *     1   0
         *       +
         *
         */
        std::vector<test_t> test{
            {0, 0, {0.0, 0.0}, {sqrt(2.0), -sqrt(2.0)}},
            {0, 1, {0.0, 0.0}, {-sqrt(2.0), -sqrt(2.0)}},
            {0, 2, {1.0, 0.0}, {0.0, 2.0 * sqrt(2.0)}},
            {1, 0, {1.0, 0.0}, {0.0, -2.0 * sqrt(2.0)}},
            {1, 1, {0.0, 0.0}, {sqrt(2.0), sqrt(2.0)}},
            {1, 2, {0.0, 0.0}, {-sqrt(2.0), sqrt(2.0)}},
        };
        for (auto& [eleNo, faceNo, xi, refN] : test) {
            auto gradE = cl.evaluateGradientAt({xi});
            auto J = Managed(cl.jacobianResultInfo(1u));
            auto JInv = Managed(cl.jacobianResultInfo(1u));
            auto detJ = Managed(cl.detJResultInfo(1u));
            auto n = Managed(cl.normalResultInfo(1u));
            cl.jacobian(eleNo, gradE, J);
            cl.jacobianInv(J, JInv);
            cl.detJ(eleNo, J, detJ);
            cl.normal(faceNo, detJ, JInv, n);
            for (std::size_t d = 0; d < 2; ++d) {
                CHECK(n(d, 0) == doctest::Approx(refN[d]));
            }
        }
    }
}

TEST_CASE("Simplex distance") {
    SUBCASE("Line") {
        auto d = SimplexDistance<1u>(Simplex<1u>::referenceSimplexVertices());
        {
            auto [dist, x] = d.closest({-1.0});
            CHECK(dist == doctest::Approx(1.0));
            CHECK(x[0] == doctest::Approx(0.0));
        }
        {
            auto [dist, x] = d.closest({3.0});
            CHECK(dist == doctest::Approx(2.0));
            CHECK(x[0] == doctest::Approx(1.0));
        }
        {
            auto [dist, x] = d.closest({0.5});
            CHECK(dist == doctest::Approx(0.0));
            CHECK(x[0] == doctest::Approx(0.5));
        }
    }
    SUBCASE("Triangle") {
        auto d = SimplexDistance<2u>(Simplex<2u>::referenceSimplexVertices());
        {
            auto [dist, x] = d.closest({-1.0, 0.0});
            CHECK(dist == doctest::Approx(1.0));
            CHECK(x[0] == doctest::Approx(0.0));
            CHECK(x[1] == doctest::Approx(0.0));
        }
        {
            auto [dist, x] = d.closest({1.5, -1.0});
            CHECK(dist == doctest::Approx(sqrt(1.25)));
            CHECK(x[0] == doctest::Approx(1.0));
            CHECK(x[1] == doctest::Approx(0.0));
        }
        {
            auto [dist, x] = d.closest({2.0, 2.0});
            CHECK(dist == doctest::Approx(sqrt(4.5)));
            CHECK(x[0] == doctest::Approx(0.5));
            CHECK(x[1] == doctest::Approx(0.5));
        }
        {
            auto [dist, x] = d.closest({0.1, 0.2});
            CHECK(dist == doctest::Approx(0.0));
            CHECK(x[0] == doctest::Approx(0.1));
            CHECK(x[1] == doctest::Approx(0.2));
        }
    }
    SUBCASE("Tet") {
        auto d = SimplexDistance<3u>(Simplex<3u>::referenceSimplexVertices());
        {
            auto [dist, x] = d.closest({0.1, 0.2, 0.3});
            CHECK(dist == doctest::Approx(0.0));
            CHECK(x[0] == doctest::Approx(0.1));
            CHECK(x[1] == doctest::Approx(0.2));
            CHECK(x[2] == doctest::Approx(0.3));
        }
        {
            auto [dist, x] = d.closest({2.0, 2.0, 2.0});
            CHECK(dist == doctest::Approx((2 - 1.0 / 3.0) * sqrt(3.0)));
            CHECK(x[0] == doctest::Approx(1.0 / 3.0));
            CHECK(x[1] == doctest::Approx(1.0 / 3.0));
            CHECK(x[2] == doctest::Approx(1.0 / 3.0));
        }
    }
}

TEST_CASE("Point locator") {
    constexpr std::size_t D = 2;
    std::array<uint64_t, 2> N = {2, 2};
    auto transform = [](std::array<double, D> const& x) -> std::array<double, D> {
        double r = 0.5 * (x[0] + 1.0);
        double phi = 0.5 * M_PI * x[1];
        return {r * cos(phi), r * sin(phi)};
    };

    std::array<std::pair<BC, BC>, D> BCs;
    BCs.fill(std::make_pair(BC::None, BC::None));
    GenMesh<D> meshGen(N, BCs);
    auto globalMesh = meshGen.uniformMesh();
    auto mesh = globalMesh->getLocalMesh();
    auto cl = std::make_shared<Curvilinear<D>>(*mesh, transform);
    auto pl = PointLocator(cl);

    auto X = Managed(cl->mapResultInfo(1));
    {
        constexpr std::size_t original_mesh_id = 6;
        constexpr auto original_xi = std::array<double, 2>{0.1, 0.2};
        auto E = cl->evaluateBasisAt({original_xi});
        cl->map(original_mesh_id, E, X);

        auto [mesh_id, xi] = pl.locate({X(0, 0), X(1, 0)});
        REQUIRE(mesh_id == original_mesh_id);

        for (std::size_t d = 0; d < D; ++d) {
            CHECK(xi[d] == doctest::Approx(original_xi[d]));
        }
    }
}
