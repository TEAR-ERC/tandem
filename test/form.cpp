#include "basis/Nodal.h"
#include "basis/WarpAndBlend.h"
#include "form/RefElement.h"
#include "quadrules/AutoRule.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "tensor/EigenMap.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"

#include "doctest.h"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

using namespace tndm;

TEST_CASE("Reference element") {
    std::array<double, 20> refModalMassMatrix = {
        .16666666666666666667,    .16666666666666666667e-1, .50000000000000000000e-1,
        .10000000000000000000,    .47619047619047619048e-2, .79365079365079365079e-2,
        .11904761904761904762e-1, .23809523809523809524e-1, .35714285714285714286e-1,
        .71428571428571428571e-1, .19841269841269841270e-2, .27777777777777777778e-2,
        .37037037037037037037e-2, .46296296296296296296e-2, .61728395061728395062e-2,
        .92592592592592592593e-2, .13888888888888888889e-1, .18518518518518518519e-1,
        .27777777777777777778e-1, .55555555555555555556e-1};

    SUBCASE("Mass Matrix") {
        for (unsigned degree = 0; degree <= 3; ++degree) {
            auto M = ModalRefElement<3>(degree).massMatrix();
            for (std::ptrdiff_t i = 0; i < M.shape(0); ++i) {
                for (std::ptrdiff_t j = 0; j < M.shape(1); ++j) {
                    if (i == j) {
                        CHECK(M(i, j) == doctest::Approx(refModalMassMatrix[i]));
                    } else {
                        CHECK(M(i, j) == doctest::Approx(0.0));
                    }
                }
            }
        }
    }

    SUBCASE("Inverse Matrix") {
        for (unsigned degree = 0; degree <= 3; ++degree) {
            auto Minv = ModalRefElement<3>(degree).inverseMassMatrix();
            for (std::ptrdiff_t i = 0; i < Minv.shape(0); ++i) {
                for (std::ptrdiff_t j = 0; j < Minv.shape(1); ++j) {
                    if (i == j) {
                        CHECK(Minv(i, j) == doctest::Approx(1.0 / refModalMassMatrix[i]));
                    } else {
                        CHECK(Minv(i, j) == doctest::Approx(0.0));
                    }
                }
            }
        }
    }
}

TEST_CASE("Projection") {
    constexpr unsigned degree = 3;
    const auto test_fun = [](double x) { return x * x * x - 4.0 * x * x + 5.0 * x - 1.0; };

    auto rule = simplexQuadratureRule<1u>(2 * degree);
    auto space = NodalRefElement(degree, WarpAndBlendFactory<1u>());
    auto Minv = space.inverseMassMatrix();
    auto E_Q = space.evaluateBasisAt(rule.points());

    auto F_Q = Managed<Vector<double>>(rule.size());
    auto F_Q_w = Managed<Vector<double>>(rule.size());
    for (unsigned q = 0, Nq = rule.size(); q < Nq; ++q) {
        F_Q(q) = test_fun(rule.points()[q][0]);
        F_Q_w(q) = rule.weights()[q] * F_Q(q);
    }

    auto F = Managed<Vector<double>>(space.numBasisFunctions());
    EigenMap(F) = EigenMap(Minv) * EigenMap(E_Q) * EigenMap(F_Q_w);

    for (unsigned k = 0, Nbf = space.numBasisFunctions(); k < Nbf; ++k) {
        CHECK(F(k) == doctest::Approx(test_fun(space.refNodes()[k][0])));
    }

    auto F_Q_test = Managed<Vector<double>>(rule.size());
    auto E_Q_T = space.evaluateBasisAt(rule.points(), {1, 0});
    EigenMap(F_Q_test) = EigenMap(E_Q_T) * EigenMap(F);

    for (unsigned q = 0, Nq = rule.size(); q < Nq; ++q) {
        CHECK(F_Q(q) == doctest::Approx(F_Q_test(q)));
    }
}
