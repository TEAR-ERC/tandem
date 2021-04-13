#include "RefElement.h"
#include "basis/Functions.h"
#include "basis/Nodal.h"
#include "quadrules/AutoRule.h"
#include "tensor/EigenMap.h"
#include "tensor/Reshape.h"
#include "tensor/TensorBase.h"
#include "util/Combinatorics.h"
#include "util/Enumerate.h"
#include "util/MultiIndex.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <cassert>
#include <cstddef>

namespace tndm {

template <std::size_t D> Managed<Matrix<double>> ModalRefElement<D>::massMatrix() const {
    auto rule = simplexQuadratureRule<D>(2 * this->degree());
    std::ptrdiff_t nbf = this->numBasisFunctions();
    Managed<Matrix<double>> M({nbf, nbf}, this->alignment());
    auto E = evaluateBasisAt(rule.points(), {0, 1});
    for (std::ptrdiff_t i = 0; i < M.shape(0); ++i) {
        for (std::ptrdiff_t j = 0; j < M.shape(1); ++j) {
            M(i, j) = 0.0;
            // The basis is orthogonal, therefore we only need to compute the diagonal
            if (i == j) {
                for (std::size_t q = 0; q < rule.size(); ++q) {
                    M(i, j) += rule.weights()[q] * E(i, q) * E(j, q);
                }
            }
        }
    }

    return M;
}

template <std::size_t D> Managed<Matrix<double>> ModalRefElement<D>::inverseMassMatrix() const {
    auto Minv = massMatrix();
    for (std::ptrdiff_t i = 0; i < Minv.shape(0); ++i) {
        Minv(i, i) = 1.0 / Minv(i, i);
    }
    return Minv;
}

template <std::size_t D>
Managed<Matrix<double>>
ModalRefElement<D>::evaluateBasisAt(std::vector<std::array<double, D>> const& points,
                                    std::array<unsigned, 2> const& permutation) const {
    using index_t = Matrix<double>::index_t;
    auto shape =
        permute(permutation, make_index<index_t>(this->numBasisFunctions(), points.size()));
    Managed<Matrix<double>> E(shape, this->alignment());
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
        permute(permutation, make_index<index_t>(this->numBasisFunctions(), D, points.size()));
    Managed<Tensor<double, 3u>> grad(shape, this->alignment());
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
NodalRefElement<D>::NodalRefElement(unsigned degree, NodesFactory<D> const& nodesFactory,
                                    std::size_t alignment)
    : RefElement<D>(degree, alignment), refNodes_(nodesFactory(degree)) {
    assert(this->numBasisFunctions() == refNodes_.size());
    vandermonde_ = Vandermonde(this->degree(), refNodes_);
    vandermondeInv_ = vandermonde_.inverse();
}

template <std::size_t D> Managed<Matrix<double>> NodalRefElement<D>::massMatrix() const {
    std::ptrdiff_t nbf = this->numBasisFunctions();
    Managed<Matrix<double>> M({nbf, nbf}, this->alignment());

    Managed<Matrix<double>> modalM =
        ModalRefElement<D>(this->degree(), this->alignment()).massMatrix();
    EigenMap(M) = vandermondeInv_.transpose() * EigenMap(modalM) * vandermondeInv_;

    return M;
}

template <std::size_t D> Managed<Matrix<double>> NodalRefElement<D>::inverseMassMatrix() const {
    std::ptrdiff_t nbf = this->numBasisFunctions();
    Managed<Matrix<double>> Minv({nbf, nbf}, this->alignment());

    Managed<Matrix<double>> modalMinv =
        ModalRefElement<D>(this->degree(), this->alignment()).inverseMassMatrix();
    EigenMap(Minv) = vandermonde_ * EigenMap(modalMinv) * vandermonde_.transpose();

    return Minv;
}

template <std::size_t D>
Managed<Matrix<double>>
NodalRefElement<D>::evaluateBasisAt(std::vector<std::array<double, D>> const& points,
                                    std::array<unsigned, 2> const& permutation) const {
    Managed<Matrix<double>> E =
        ModalRefElement<D>(this->degree(), this->alignment()).evaluateBasisAt(points, permutation);
    auto Emap = EigenMap(E);
    if (permutation[0] == 0 && permutation[1] == 1) {
        Emap = vandermondeInv_.transpose() * Emap;
    } else if (permutation[0] == 1 && permutation[1] == 0) {
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
    Managed<Tensor<double, 3u>> gradE = ModalRefElement<D>(this->degree(), this->alignment())
                                            .evaluateGradientAt(points, permutation);

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

template class ModalRefElement<1ul>;
template class ModalRefElement<2ul>;
template class ModalRefElement<3ul>;
template class NodalRefElement<1ul>;
template class NodalRefElement<2ul>;
template class NodalRefElement<3ul>;

} // namespace tndm
