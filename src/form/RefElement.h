#ifndef REFELEMENT_20200618_H
#define REFELEMENT_20200618_H

#include "basis/Nodal.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/Combinatorics.h"

#include <Eigen/Core>

#include <array>
#include <memory>
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
    virtual std::unique_ptr<RefElement<D>> clone() const = 0;
    /**
     * @brief Evaluate basis functions at points
     *
     * @param points Evaluation points (e.g. quadrature points)
     *
     * @return Matrix with shape (numberOfBasisFunctions, numberOfPoints)
     */
    Managed<Matrix<double>>
    evaluateBasisAt(std::vector<std::array<double, D>> const& points) const {
        return evaluateBasisAt(points, {0, 1});
    }
    /**
     * @brief evaluateBasisAt with output permutation
     */
    virtual Managed<Matrix<double>>
    evaluateBasisAt(std::vector<std::array<double, D>> const& points,
                    std::array<unsigned, 2> const& permutation) const = 0;
    /**
     * @brief Evaluate gradient of basis functions at points
     *
     * @param points Evaluation points (e.g. quadrature points)
     *
     * @return Tensor with shape (numberOfBasisFunctions, D, numberOfPoints)
     */
    Managed<Tensor<double, 3u>>
    evaluateGradientAt(std::vector<std::array<double, D>> const& points) const {
        return evaluateGradientAt(points, {0, 1, 2});
    }
    /**
     * @brief evaluateGradientAt with output permutation
     */
    virtual Managed<Tensor<double, 3u>>
    evaluateGradientAt(std::vector<std::array<double, D>> const& points,
                       std::array<unsigned, 3> const& permutation) const = 0;

    unsigned degree() const { return degree_; }
    std::size_t numberOfBasisFunctions() const { return binom(degree_ + D, D); }

private:
    unsigned degree_;
};

template <std::size_t D> class ModalRefElement : public RefElement<D> {
public:
    using RefElement<D>::RefElement;
    using RefElement<D>::evaluateBasisAt;
    using RefElement<D>::evaluateGradientAt;

    std::unique_ptr<RefElement<D>> clone() const override {
        return std::make_unique<ModalRefElement<D>>(*this);
    }

    Managed<Matrix<double>>
    evaluateBasisAt(std::vector<std::array<double, D>> const& points,
                    std::array<unsigned, 2> const& permutation) const override;

    Managed<Tensor<double, 3u>>
    evaluateGradientAt(std::vector<std::array<double, D>> const& points,
                       std::array<unsigned, 3> const& permutation) const override;
};

template <std::size_t D> class NodalRefElement : public RefElement<D> {
public:
    using RefElement<D>::evaluateBasisAt;
    using RefElement<D>::evaluateGradientAt;

    NodalRefElement(unsigned degree, NodesFactory<D> const& nodesFactory);
    std::unique_ptr<RefElement<D>> clone() const override {
        return std::make_unique<NodalRefElement<D>>(*this);
    }

    Managed<Matrix<double>>
    evaluateBasisAt(std::vector<std::array<double, D>> const& points,
                    std::array<unsigned, 2> const& permutation) const override;

    Managed<Tensor<double, 3u>>
    evaluateGradientAt(std::vector<std::array<double, D>> const& points,
                       std::array<unsigned, 3> const& permutation) const override;

    auto const& refNodes() const { return refNodes_; }

private:
    std::vector<std::array<double, D>> refNodes_;
    Eigen::MatrixXd vandermondeInv_;
};

} // namespace tndm

#endif // REFELEMENT_20200618_H
