#include "FiniteElementFunction.h"
#include "form/RefElement.h"
#include "tensor/EigenMap.h"

#include <Eigen/Core>

#include <cassert>

namespace tndm {

template <std::size_t D>
Managed<Matrix<double>>
FiniteElementFunction<D>::evaluationMatrix(std::vector<std::array<double, D>> const& points) const {
    return refElement_->evaluateBasisAt(points, {1, 0});
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

template class FiniteElementFunction<2ul>;
template class FiniteElementFunction<3ul>;

} // namespace tndm
