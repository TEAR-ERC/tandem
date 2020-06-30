#include "RefElement.h"
#include "basis/Functions.h"
#include "tensor/EigenMap.h"
#include "tensor/Reshape.h"
#include "util/Combinatorics.h"
#include "util/Enumerate.h"
#include "util/MultiIndex.h"

#include <Eigen/LU>

namespace tndm {

template <std::size_t D>
Managed<Matrix<double>>
ModalRefElement<D>::evaluateBasisAt(std::vector<std::array<double, D>> const& points,
                                    std::array<unsigned, 2> const& permutation) const {
    using index_t = Matrix<double>::index_t;
    auto shape =
        permute(permutation, make_index<index_t>(this->numberOfBasisFunctions(), points.size()));
    Managed<Matrix<double>> E(shape);
    for (std::size_t p = 0; p < points.size(); ++p) {
        for (auto&& [bf, j] : enumerate(AllIntegerSums<D>(this->degree()))) {
            auto index = permute(permutation, make_index<index_t>(bf, p));
            E(index) = DubinerP(j, points[p]);
        }
    }
    return E;
}

template <std::size_t D>
Managed<Tensor<double, 3u>>
ModalRefElement<D>::evaluateGradientAt(std::vector<std::array<double, D>> const& points,
                                       std::array<unsigned, 3> const& permutation) const {
    using index_t = Matrix<double>::index_t;
    auto shape =
        permute(permutation, make_index<index_t>(this->numberOfBasisFunctions(), D, points.size()));
    Managed<Tensor<double, 3u>> grad(shape);
    for (std::size_t p = 0; p < points.size(); ++p) {
        for (auto&& [bf, j] : enumerate(AllIntegerSums<D>(this->degree()))) {
            auto dphi = gradDubinerP(j, points[p]);
            for (std::size_t d = 0; d < D; ++d) {
                auto index = permute(permutation, make_index<index_t>(bf, d, p));
                grad(index) = dphi[d];
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
    vandermondeInv_ = vandermonde.inverse();
}

template <std::size_t D>
Managed<Matrix<double>>
NodalRefElement<D>::evaluateBasisAt(std::vector<std::array<double, D>> const& points,
                                    std::array<unsigned, 2> const& permutation) const {
    Managed<Matrix<double>> E =
        ModalRefElement<D>(this->degree()).evaluateBasisAt(points, permutation);
    auto Emap = EigenMap(E);
    if (permutation[0] == 0 && permutation[1] == 1) {
        Emap = vandermondeInv_.transpose() * Emap;
    } else if (permutation[0] == 1 && permutation[0] == 0) {
        Emap = Emap * vandermondeInv_;
    } else {
        assert(false);
    }
    return E;
}

template <std::size_t D>
Managed<Tensor<double, 3u>>
NodalRefElement<D>::evaluateGradientAt(std::vector<std::array<double, D>> const& points,
                                       std::array<unsigned, 3> const& permutation) const {
    Managed<Tensor<double, 3u>> gradE =
        ModalRefElement<D>(this->degree()).evaluateGradientAt(points, permutation);

    assert(vandermondeInv_.cols() == vandermondeInv_.rows());
    // 0,1,2 F_idq = V_ji E_jdq => F_i(dq) = V^T E_j(dq)
    // 0,2,1 F_iqd = V_ji E_jqd => F_i(qd) = V^T E_j(qd)
    if (permutation[0] == 0) {
        assert((permutation[1] == 1 && permutation[2] == 2) ||
               (permutation[1] == 2 && permutation[2] == 1));
        auto mat = reshape(gradE, vandermondeInv_.rows(), D * points.size());
        EigenMap(mat) = vandermondeInv_.transpose() * EigenMap(mat);
    }
    // 1,2,0 F_dqi = V_ji E_dqj => F_(dq)i = E_(dq)j V
    // 2,1,0 F_qdi = V_ji E_qdj => F_(qd)i = E_(qd)j V
    else if (permutation[2] == 0) {
        assert((permutation[0] == 1 && permutation[1] == 2) ||
               (permutation[0] == 2 && permutation[1] == 1));
        auto mat = reshape(gradE, D * points.size(), vandermondeInv_.rows());
        EigenMap(mat) = EigenMap(mat) * vandermondeInv_;
    }
    // 1,0,2 F_diq = V_ji E_djq => F_di[q] = E_dj[q] V
    // 2,0,1 F_qid = V_ji E_qjd => F_qi[d] = E_qj[d] V
    else {
        assert((permutation[0] == 1 && permutation[2] == 2) ||
               (permutation[0] == 2 && permutation[2] == 1));
        for (decltype(gradE)::index_t i = 0; i < gradE.shape(2); ++i) {
            auto mat = gradE.subtensor(slice{}, slice{}, i);
            EigenMap(mat) = EigenMap(mat) * vandermondeInv_;
        }
    }

    return gradE;
}

template class ModalRefElement<2ul>;
template class ModalRefElement<3ul>;
template class NodalRefElement<2ul>;
template class NodalRefElement<3ul>;

} // namespace tndm
