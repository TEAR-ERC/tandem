#ifndef REFELEMENT_20200618_H
#define REFELEMENT_20200618_H

#include "basis/Functions.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/Combinatorics.h"
#include "util/Enumerate.h"

#include <array>
#include <vector>

namespace tndm {

template <std::size_t D>
auto dubinerBasisAt(unsigned maxDegree, std::vector<std::array<double, D>> const& points) {
    auto Nbf = binom(maxDegree + D, D);
    Managed<Matrix<double>> E(Nbf, points.size());
    for (std::size_t p = 0; p < points.size(); ++p) {
        for (auto&& [bf, j] : enumerate(AllIntegerSums<D>(maxDegree))) {
            E(bf, p) = DubinerP(j, points[p]);
        }
    }
    return E;
}

template <std::size_t D>
auto dubinerBasisGradientAt(unsigned maxDegree, std::vector<std::array<double, D>> const& points) {
    auto Nbf = binom(maxDegree + D, D);
    Managed<Tensor<double, 3u>> grad(Nbf, D, points.size());
    for (std::size_t p = 0; p < points.size(); ++p) {
        for (auto&& [bf, j] : enumerate(AllIntegerSums<D>(maxDegree))) {
            auto dphi = gradDubinerP(j, points[p]);
            for (std::size_t d = 0; d < D; ++d) {
                grad(bf, d, p) = dphi[d];
            }
        }
    }
    return grad;
}

} // namespace tndm

#endif // REFELEMENT_20200618_H
