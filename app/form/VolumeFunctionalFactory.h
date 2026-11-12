#ifndef VOLUMEFUNCTIONALFACTORY_20210910_H
#define VOLUMEFUNCTIONALFACTORY_20210910_H

#include "tensor/Tensor.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

namespace tndm {

class AbstractVolumeFunctionalFactory {
public:
    using volume_functional_t = std::function<void(std::size_t, Matrix<double>&)>;

    virtual ~AbstractVolumeFunctionalFactory() {}
    virtual auto operator()() const -> volume_functional_t = 0;
};

template <typename LocalOperator>
class VolumeFunctionalFactory : public AbstractVolumeFunctionalFactory {
public:
    static constexpr std::size_t NumQuantities = LocalOperator::NumQuantities;
    using functional_t = std::function<std::array<double, NumQuantities>(
        std::array<double, LocalOperator::Dim> const&, long int&)>;

    VolumeFunctionalFactory(std::shared_ptr<LocalOperator> lop, functional_t function)
        : lop_(std::move(lop)), function_(std::move(function)) {}

    auto operator()() const -> volume_functional_t override {
        return lop_->make_volume_functional(function_);
    }

private:
    std::shared_ptr<LocalOperator> lop_;
    functional_t function_;
};

} // namespace tndm

#endif // VOLUMEFUNCTIONALFACTORY_20210910_H
