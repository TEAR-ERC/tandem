#include "FiniteElementFunction.h"
#include "form/RefElement.h"
#include "tensor/EigenMap.h"

#include <Eigen/Core>

#include <cassert>
#include <cstdio>

namespace tndm {

template <std::size_t D>
Managed<Matrix<double>>
FiniteElementFunction<D>::evaluationMatrix(std::vector<std::array<double, D>> const& points) const {
    return refElement_->evaluateBasisAt(points, {1, 0});
}

template <std::size_t D>
Managed<Tensor<double, 3u>> FiniteElementFunction<D>::gradientEvaluationTensor(
    std::vector<std::array<double, D>> const& points) const {
    return refElement_->evaluateGradientAt(points, {2, 0, 1});
}

template <std::size_t D>
TensorBase<Matrix<double>> FiniteElementFunction<D>::mapResultInfo(std::size_t numPoints) const {
    return TensorBase<Matrix<double>>(numPoints, numQuantities());
}

template <std::size_t D>
void FiniteElementFunction<D>::map(std::size_t eleNo, Matrix<double> const& evalMatrix,
                                   Tensor<double, 2u>& result) const {
    assert(eleNo < numElements());
    assert(result.shape(0) == evalMatrix.shape(0));
    assert(result.shape(1) == numQuantities());
    assert(evalMatrix.shape(1) == numBasisFunctions());

    auto mat = data_.subtensor(slice{}, slice{}, eleNo);
    EigenMap(result) = EigenMap(evalMatrix) * EigenMap(mat);
}

template <std::size_t D>
TensorBase<Tensor<double, 3u>>
FiniteElementFunction<D>::gradientResultInfo(std::size_t numPoints) const {
    return TensorBase<Tensor<double, 3u>>(numPoints, numQuantities(), D);
}

template <std::size_t D>
void FiniteElementFunction<D>::gradient(std::size_t eleNo, Tensor<double, 3u> const& evalTensor,
                                        Tensor<double, 3u> const& jInvAtP,
                                        Tensor<double, 3u>& result) const {
    assert(eleNo < numElements());
    assert(result.shape(0) == evalTensor.shape(0));
    assert(result.shape(1) == numQuantities());
    assert(result.shape(2) == D);
    assert(evalTensor.shape(1) == numBasisFunctions());
    assert(evalTensor.shape(2) == D);
    assert(jInvAtP.shape(0) == D);
    assert(jInvAtP.shape(1) == D);
    assert(jInvAtP.shape(2) == result.shape(0));

    auto mat = data_.subtensor(slice{}, slice{}, eleNo);
    for (std::ptrdiff_t j = 0; j < evalTensor.shape(2); ++j) {
        auto d_j = evalTensor.subtensor(slice{}, slice{}, j);
        auto result_j = result.subtensor(slice{}, slice{}, j);
        EigenMap(result_j) = EigenMap(d_j) * EigenMap(mat);
    }

    for (std::ptrdiff_t q = 0; q < result.shape(0); ++q) {
        auto jInvAtP_q = jInvAtP.subtensor(slice{}, slice{}, q);
        auto result_q = result.subtensor(q, slice{}, slice{});
        EigenMap(result_q) = EigenMap(result_q) * EigenMap(jInvAtP_q);
    }
}

template <std::size_t D> std::string FiniteElementFunction<D>::name(std::size_t q) const {
    assert(q < numQuantities());
    if (!names_.empty()) {
        return names_[q];
    }
    char buf[100];
    snprintf(buf, sizeof(buf), "q%lu", q);
    return std::string(buf);
}

template class FiniteElementFunction<1ul>;
template class FiniteElementFunction<2ul>;
template class FiniteElementFunction<3ul>;

} // namespace tndm
