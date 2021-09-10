#ifndef FACETFUNCTIONALFACTORY_20210910_H
#define FACETFUNCTIONALFACTORY_20210910_H

#include "tensor/Tensor.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

namespace tndm {

class AbstractFacetFunctionalFactory {
public:
    using facet_functional_t = std::function<void(std::size_t, Matrix<double>&, bool)>;

    virtual ~AbstractFacetFunctionalFactory() {}
    virtual auto operator()(double time) const -> facet_functional_t = 0;
};

template <typename LocalOperator>
class FacetFunctionalFactory : public AbstractFacetFunctionalFactory {
public:
    static constexpr std::size_t NumQuantities = LocalOperator::NumQuantities;
    using time_functional_t = std::function<std::array<double, NumQuantities>(
        std::array<double, LocalOperator::Dim + 1u> const&)>;

    FacetFunctionalFactory(std::shared_ptr<LocalOperator> lop, time_functional_t time_function,
                           std::array<double, LocalOperator::Dim> const& ref_normal)
        : lop_(std::move(lop)), time_function_(std::move(time_function)), ref_normal_(ref_normal) {}

    auto operator()(double time) const -> facet_functional_t override {
        auto function =
            std::function([this, time](std::array<double, LocalOperator::Dim> const& x) {
                std::array<double, LocalOperator::Dim + 1u> xt;
                std::copy(x.begin(), x.end(), xt.begin());
                xt.back() = time;
                return time_function_(xt);
            });
        return lop_->make_facet_functional(std::move(function), ref_normal_);
    }

private:
    std::shared_ptr<LocalOperator> lop_;
    time_functional_t time_function_;
    std::array<double, LocalOperator::Dim> ref_normal_;
};

} // namespace tndm

#endif // FACETFUNCTIONALFACTORY_20210910_H
