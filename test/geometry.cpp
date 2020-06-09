#include "geometry/Curvilinear.h"
#include "mesh/GenMesh.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"

#include "doctest.h"

#include <array>
#include <cmath>
#include <tuple>
#include <vector>

using tndm::Curvilinear;
using tndm::GenMesh;
using tndm::Managed;
using tndm::Tensor;

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

    GenMesh<D> meshGen(N);
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
            auto JinvT = Managed(cl.jacobianResultInfo(1u));
            auto detJ = Managed(cl.detJResultInfo(1u));
            auto n = Managed(cl.normalResultInfo(1u));
            cl.jacobian(eleNo, gradE, J);
            cl.jacobianInvT(J, JinvT);
            cl.detJ(eleNo, J, detJ);
            cl.normal(faceNo, detJ, JinvT, n);
            for (std::size_t d = 0; d < 2; ++d) {
                CHECK(n(d, 0) == doctest::Approx(refN[d]));
            }
        }
    }
}
