#include "doctest.h"

#include "form/RefElement.h"

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
