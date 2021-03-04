#ifndef REFELEMENT_20200618_H
#define REFELEMENT_20200618_H

#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/Combinatorics.h"

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

template <std::size_t D> class NodesFactory;

/**
 * @brief Function space in reference element
 *
 * @tparam D Domain dimension
 */
template <std::size_t D> class RefElement {
public:
    static constexpr std::size_t DefaultAlignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;

    /**
     * @brief Constructor
     *
     * @param degree Maximum polynomial degree
     * @param alignment Tensors returned by this class will respect this memory alignment.
     */
    RefElement(unsigned degree, std::size_t alignment = DefaultAlignment)
        : degree_(degree), alignment_(alignment) {}
    virtual ~RefElement() {}
    virtual std::unique_ptr<RefElement<D>> clone() const = 0;

    /**
     * @brief Computes int phi_i phi_j dV
     *
     * Only for linear elements!
     */
    virtual Managed<Matrix<double>> massMatrix() const = 0;
    virtual Managed<Matrix<double>> inverseMassMatrix() const = 0;

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
    std::size_t alignment() const { return alignment_; }
    std::size_t numBasisFunctions() const { return binom(degree_ + D, D); }

private:
    unsigned degree_;
    std::size_t alignment_;
};

template <std::size_t D> class ModalRefElement : public RefElement<D> {
public:
    using RefElement<D>::RefElement;
    using RefElement<D>::evaluateBasisAt;
    using RefElement<D>::evaluateGradientAt;

    std::unique_ptr<RefElement<D>> clone() const override {
        return std::make_unique<ModalRefElement<D>>(*this);
    }

    Managed<Matrix<double>> massMatrix() const override;
    Managed<Matrix<double>> inverseMassMatrix() const override;

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

    NodalRefElement(unsigned degree, NodesFactory<D> const& nodesFactory,
                    std::size_t alignment = RefElement<D>::DefaultAlignment);
    std::unique_ptr<RefElement<D>> clone() const override {
        return std::make_unique<NodalRefElement<D>>(*this);
    }

    Managed<Matrix<double>> massMatrix() const override;
    Managed<Matrix<double>> inverseMassMatrix() const override;

    Managed<Matrix<double>>
    evaluateBasisAt(std::vector<std::array<double, D>> const& points,
                    std::array<unsigned, 2> const& permutation) const override;

    Managed<Tensor<double, 3u>>
    evaluateGradientAt(std::vector<std::array<double, D>> const& points,
                       std::array<unsigned, 3> const& permutation) const override;

    auto const& refNodes() const { return refNodes_; }

private:
    std::vector<std::array<double, D>> refNodes_;
    Eigen::MatrixXd vandermonde_, vandermondeInv_;
};

} // namespace tndm

#endif // REFELEMENT_20200618_H
