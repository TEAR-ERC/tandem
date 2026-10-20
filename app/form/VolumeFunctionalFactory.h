#ifndef VOLUMEFUNCTIONALFACTORY_20210910_H
#define VOLUMEFUNCTIONALFACTORY_20210910_H

#include "tensor/Tensor.h"

#include <algorithm>
#include <array>
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
        std::array<double, LocalOperator::Dim> const&)>;

    VolumeFunctionalFactory(std::shared_ptr<LocalOperator> lop, functional_t function)
        : lop_(std::move(lop)), function_(std::move(function)) {}

    auto operator()() const -> volume_functional_t override {
        return lop_->make_volume_functional(function_);
    }

private:
    std::shared_ptr<LocalOperator> lop_;
    functional_t function_;
};

// Time-dependent counterpart, for a body force that changes from step to step.
// Mirrors FacetFunctionalFactory: operator()(time) binds the time and returns the
// volume functional the operator assembles with.
class AbstractTimeVolumeFunctionalFactory {
public:
    using volume_functional_t = std::function<void(std::size_t, Matrix<double>&)>;

    virtual ~AbstractTimeVolumeFunctionalFactory() {}
    virtual auto operator()(double time) const -> volume_functional_t = 0;
};

template <typename LocalOperator>
class TimeVolumeFunctionalFactory : public AbstractTimeVolumeFunctionalFactory {
public:
    static constexpr std::size_t NumQuantities = LocalOperator::NumQuantities;
    using time_functional_t = std::function<std::array<double, NumQuantities>(
        std::array<double, LocalOperator::Dim + 1u> const&)>;

    TimeVolumeFunctionalFactory(std::shared_ptr<LocalOperator> lop, time_functional_t time_function)
        : lop_(std::move(lop)), time_function_(std::move(time_function)) {}

    auto operator()(double time) const -> volume_functional_t override {
        auto function =
            std::function([this, time](std::array<double, LocalOperator::Dim> const& x) {
                std::array<double, LocalOperator::Dim + 1u> xt;
                std::copy(x.begin(), x.end(), xt.begin());
                xt.back() = time;
                return time_function_(xt);
            });
        return lop_->make_volume_functional(std::move(function));
    }

private:
    std::shared_ptr<LocalOperator> lop_;
    time_functional_t time_function_;
};

} // namespace tndm

#endif // VOLUMEFUNCTIONALFACTORY_20210910_H
