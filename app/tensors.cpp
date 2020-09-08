#include "basis/Functions.h"
#include "form/RefElement.h"
#include "quadrules/AutoRule.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "tensor/EigenMap.h"
#include "tensor/Managed.h"
#include "util/Combinatorics.h"

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using Eigen::DiagonalMatrix;
using Eigen::MatrixXd;
using tndm::AllIntegerSums;
using tndm::binom;
using tndm::gradTetraDubinerP;
using tndm::ModalRefElement;
using tndm::TetraDubinerP;

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: `tensors <degree>`" << std::endl;
        return -1;
    }

    unsigned N = atoi(argv[1]);
    unsigned numBF = binom(N + 3, 3);

    std::cout << "Number of basis functions: " << numBF << std::endl;

    auto rule = tndm::simplexQuadratureRule<3u>(2u * N + 1u);

    auto truncate = [](double x) { return (std::fabs(x) < 1e-15) ? 0.0 : x; };

    DiagonalMatrix<double, Eigen::Dynamic> W(rule.size());

    auto phi = ModalRefElement<3u>(N).evaluateBasisAt(rule.points());
    auto phiMap = EigenMap(phi);

    for (std::size_t i = 0; i < rule.size(); ++i) {
        W.diagonal()(i) = rule.weights()[i];
    }

    auto m = phiMap * W * phiMap.transpose();
    std::cout << "Mass matrix:" << std::endl;
    std::cout << m.unaryExpr(truncate) << std::endl;

    std::array<MatrixXd, 3> dphi;
    for (auto& matrix : dphi) {
        matrix.resize(rule.size(), numBF);
    }
    std::size_t bf = 0;
    for (auto j : AllIntegerSums<3>(N)) {
        for (std::size_t i = 0; i < rule.size(); ++i) {
            auto grad = gradTetraDubinerP(j, rule.points()[i]);
            for (std::size_t d = 0; d < grad.size(); ++d) {
                dphi[d](i, bf) = grad[d];
            }
        }
        ++bf;
    }

    auto kXi = dphi[0].transpose() * W * phiMap.transpose();
    std::cout << "dphidxi * phi:" << std::endl;
    std::cout << kXi.unaryExpr(truncate) << std::endl;

    for (std::size_t x = 0; x < 3; ++x) {
        for (std::size_t y = 0; y < 3; ++y) {
            auto kxy = dphi[x].transpose() * W * dphi[y];
            std::cout << "dphi[" << x << "] * dphi[" << y << "]:" << std::endl;
            std::cout << kxy.unaryExpr(truncate) << std::endl;
        }
    }

    return 0;
}
