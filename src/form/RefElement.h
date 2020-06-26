#ifndef REFELEMENT_20200618_H
#define REFELEMENT_20200618_H

#include "basis/Nodal.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/Combinatorics.h"

#include <Eigen/Core>

#include <array>
#include <vector>

namespace tndm {

/**
 * @brief Function space in reference element
 *
 * @tparam D Domain dimension
 */
template <std::size_t D> class RefElement {
public:
    RefElement(unsigned degree) : degree_(degree) {}
    virtual ~RefElement() {}
    /**
     * @brief Evaluate basis functions at points
     *
     * @param points Evaluation points (e.g. quadrature points)
     *
     * @return Matrix with shape (numberOfBasisFunctions, numberOfPoints)
     */
    virtual Managed<Matrix<double>>
    evaluateBasisAt(std::vector<std::array<double, D>> const& points) const = 0;
    /**
     * @brief Evaluate gradient of basis functions at points
     *
     * @param points Evaluation points (e.g. quadrature points)
     *
     * @return Tensor with shape (numberOfBasisFunctions, D, numberOfPoints)
     */
    virtual Managed<Tensor<double, 3u>>
    evaluateGradientAt(std::vector<std::array<double, D>> const& points) const = 0;

    unsigned degree() const { return degree_; }
    std::size_t numberOfBasisFunctions() const { return binom(degree_ + D, D); }

private:
    unsigned degree_;
};

template <std::size_t D> class ModalRefElement : public RefElement<D> {
public:
    using RefElement<D>::RefElement;

    Managed<Matrix<double>>
    evaluateBasisAt(std::vector<std::array<double, D>> const& points) const override;

    Managed<Tensor<double, 3u>>
    evaluateGradientAt(std::vector<std::array<double, D>> const& points) const override;
};

template <std::size_t D> class NodalRefElement : public RefElement<D> {
public:
    NodalRefElement(unsigned degree, NodesFactory<D> const& nodesFactory);

    Managed<Matrix<double>>
    evaluateBasisAt(std::vector<std::array<double, D>> const& points) const override;

    Managed<Tensor<double, 3u>>
    evaluateGradientAt(std::vector<std::array<double, D>> const& points) const override;

    auto const& refNodes() const { return refNodes_; }

private:
    std::vector<std::array<double, D>> refNodes_;
    Eigen::MatrixXd vandermondeInvT_;
};

} // namespace tndm

#endif // REFELEMENT_20200618_H
