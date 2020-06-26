#include "RefElement.h"
#include "basis/Functions.h"
#include "tensor/EigenMap.h"
#include "tensor/Reshape.h"
#include "util/Combinatorics.h"
#include "util/Enumerate.h"

#include <Eigen/LU>

namespace tndm {

template <std::size_t D>
Managed<Matrix<double>>
ModalRefElement<D>::evaluateBasisAt(std::vector<std::array<double, D>> const& points) const {
    Managed<Matrix<double>> E(this->numberOfBasisFunctions(), points.size());
    for (std::size_t p = 0; p < points.size(); ++p) {
        for (auto&& [bf, j] : enumerate(AllIntegerSums<D>(this->degree()))) {
            E(bf, p) = DubinerP(j, points[p]);
        }
    }
    return E;
}

template <std::size_t D>
Managed<Tensor<double, 3u>>
ModalRefElement<D>::evaluateGradientAt(std::vector<std::array<double, D>> const& points) const {
    Managed<Tensor<double, 3u>> grad(this->numberOfBasisFunctions(), D, points.size());
    for (std::size_t p = 0; p < points.size(); ++p) {
        for (auto&& [bf, j] : enumerate(AllIntegerSums<D>(this->degree()))) {
            auto dphi = gradDubinerP(j, points[p]);
            for (std::size_t d = 0; d < D; ++d) {
                grad(bf, d, p) = dphi[d];
            }
        }
    }
    return grad;
}

template <std::size_t D>
NodalRefElement<D>::NodalRefElement(unsigned degree, NodesFactory<D> const& nodesFactory)
    : RefElement<D>(degree), refNodes_(nodesFactory(degree)) {
    assert(this->numberOfBasisFunctions() == refNodes_.size());
    auto vandermonde = Vandermonde(this->degree(), refNodes_);
    vandermondeInvT_ = vandermonde.inverse().transpose();
}

template <std::size_t D>
Managed<Matrix<double>>
NodalRefElement<D>::evaluateBasisAt(std::vector<std::array<double, D>> const& points) const {
    Managed<Matrix<double>> E = ModalRefElement<D>(this->degree()).evaluateBasisAt(points);
    auto Emap = EigenMap(E);
    Emap = vandermondeInvT_ * Emap;
    return E;
}

template <std::size_t D>
Managed<Tensor<double, 3u>>
NodalRefElement<D>::evaluateGradientAt(std::vector<std::array<double, D>> const& points) const {
    Managed<Tensor<double, 3u>> gradE = ModalRefElement<D>(this->degree()).evaluateGradientAt(points);

    assert(vandermondeInvT_.cols() == vandermondeInvT_.rows());
    auto matView = reshape(gradE, vandermondeInvT_.cols(), D * points.size());
    auto map = EigenMap(matView);
    map = vandermondeInvT_ * map;

    return gradE;
}

template class ModalRefElement<2ul>;
template class ModalRefElement<3ul>;
template class NodalRefElement<2ul>;
template class NodalRefElement<3ul>;

} // namespace tndm
