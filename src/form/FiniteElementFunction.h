#ifndef FINITEELEMENTFUNCTION_20200630_H
#define FINITEELEMENTFUNCTION_20200630_H

#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "tensor/TensorBase.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tndm {

template <std::size_t D> class RefElement;

template <std::size_t D> class FiniteElementFunction {
public:
    FiniteElementFunction(std::unique_ptr<RefElement<D>> refElement, std::vector<std::string> names,
                          std::size_t numberOfElements)
        : refElement_(std::move(refElement)), names_(std::move(names)),
          data_(refElement_->numBasisFunctions(), names_.size(), numberOfElements) {}

    FiniteElementFunction(std::unique_ptr<RefElement<D>> refElement, std::size_t numberOfQuantities,
                          std::size_t numberOfElements)
        : refElement_(std::move(refElement)),
          data_(refElement_->numBasisFunctions(), numberOfQuantities, numberOfElements) {}

    Managed<Matrix<double>>
    evaluationMatrix(std::vector<std::array<double, D>> const& points) const;

    Managed<Tensor<double, 3u>>
    gradientEvaluationTensor(std::vector<std::array<double, D>> const& points) const;

    TensorBase<Matrix<double>> mapResultInfo(std::size_t numPoints) const;
    void map(std::size_t eleNo, Matrix<double> const& evalMatrix, Tensor<double, 2u>& result) const;

    TensorBase<Tensor<double, 3u>> gradientResultInfo(std::size_t numPoints) const;
    void gradient(std::size_t eleNo, Tensor<double, 3u> const& evalTensor,
                  Tensor<double, 3u> const& jInvAtP, Tensor<double, 3u>& result) const;

    RefElement<D> const& refElement() const { return *refElement_; }
    std::size_t numBasisFunctions() const { return data_.shape(0); }
    std::size_t numQuantities() const { return data_.shape(1); }
    std::size_t numElements() const { return data_.shape(2); }

    auto& values() { return data_; }
    auto const& values() const { return data_; }
    std::string name(std::size_t q) const;

    void setNames(std::vector<std::string> names) { names_ = std::move(names); }

private:
    std::unique_ptr<RefElement<D>> refElement_;
    std::vector<std::string> names_;
    Managed<Tensor<double, 3u>> data_;
};

} // namespace tndm

#endif // FINITEELEMENTFUNCTION_20200630_H
