#include <iostream>

#include "basis/Functions.h"
#include "quadrules/TensorProductRule.h"
#include "util/Combinatorics.h"

#include <Eigen/Core>

#include <cstdlib>

using Eigen::DiagonalMatrix;
using Eigen::MatrixXd;
using tndm::AllIntegerSums;
using tndm::binom;
using tndm::gradTetraDubinerP;
using tndm::TetraDubinerP;

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: `tensors <degree>`" << std::endl;
        return -1;
    }

    unsigned N = atoi(argv[1]);
    unsigned numBF = binom(N + 3, 3);

    std::cout << "Number of basis functions: " << numBF << std::endl;

    auto rule = tndm::tensorProductRule<3u>(N + 1);

    auto truncate = [](double x) { return (std::fabs(x) < 1e-15) ? 0.0 : x; };

    MatrixXd phi(rule.size(), numBF);
    DiagonalMatrix<double, Eigen::Dynamic> W(rule.size());

    std::size_t bf = 0;
    for (auto j : AllIntegerSums<3>(N)) {
        for (std::size_t i = 0; i < rule.size(); ++i) {
            phi(i, bf) = TetraDubinerP(j, rule.points()[i]);
        }
        ++bf;
    }
    for (std::size_t i = 0; i < rule.size(); ++i) {
        W.diagonal()(i) = rule.weights()[i];
    }

    auto m = phi.transpose() * W * phi;
    std::cout << "Mass matrix:" << std::endl;
    std::cout << m.unaryExpr(truncate) << std::endl;

    std::array<MatrixXd, 3> dphi;
    for (auto& matrix : dphi) {
        matrix.resize(rule.size(), numBF);
    }
    bf = 0;
    for (auto j : AllIntegerSums<3>(N)) {
        for (std::size_t i = 0; i < rule.size(); ++i) {
            auto grad = gradTetraDubinerP(j, rule.points()[i]);
            for (std::size_t d = 0; d < grad.size(); ++d) {
                dphi[d](i, bf) = grad[d];
            }
        }
        ++bf;
    }

    auto kXi = dphi[0].transpose() * W * phi;
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
