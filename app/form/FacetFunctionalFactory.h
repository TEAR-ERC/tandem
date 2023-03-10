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
    using tagged_time_functional_t = std::function<std::array<double, NumQuantities>(
        std::array<double, LocalOperator::Dim + 2u> const&)>;

    FacetFunctionalFactory(std::shared_ptr<LocalOperator> lop, tagged_time_functional_t time_function,
                           std::array<double, LocalOperator::Dim> const& ref_normal)
        : lop_(std::move(lop)), time_function_(std::move(time_function)), ref_normal_(ref_normal) {}

    auto operator()(double time) const -> facet_functional_t override {
        auto function =
            std::function([this, time](std::array<double, LocalOperator::Dim+1u> const& args) {
                std::array<double, LocalOperator::Dim + 2u> xt;
                std::copy(args.begin(), args.end()-1, xt.begin());
                xt[LocalOperator::Dim + 0u] = time;
                xt[LocalOperator::Dim + 1u] = args.back();
                return time_function_(xt);
            });
        return lop_->make_facet_functional(std::move(function), ref_normal_);
    }

private:
    std::shared_ptr<LocalOperator> lop_;
    tagged_time_functional_t time_function_;
    std::array<double, LocalOperator::Dim> ref_normal_;
};

} // namespace tndm

#endif // FACETFUNCTIONALFACTORY_20210910_H
