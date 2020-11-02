#ifndef RATEANDSTATE_20201026_H
#define RATEANDSTATE_20201026_H

#include "config.h"
#include "tandem/RateAndStateBase.h"

#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>

namespace tndm {

template <class Law> class RateAndState : public RateAndStateBase {
public:
    using RateAndStateBase::RateAndStateBase;

    using param_fun_t =
        std::function<typename Law::Params(std::array<double, DomainDimension> const&)>;
    using source_fun_t =
        std::function<std::array<double, 1>(std::array<double, DomainDimension + 1> const&)>;

    void set_constant_params(typename Law::ConstantParams const& cps) {
        law_.set_constant_params(cps);
    }
    void set_params(param_fun_t pfun) {
        auto num_nodes = fault_.storage().size();
        law_.set_num_nodes(num_nodes);
        for (std::size_t index = 0; index < num_nodes; ++index) {
            auto params = pfun(fault_.storage()[index].template get<Coords>());
            law_.set_params(index, params);
        }
    }

    void set_source_fun(source_fun_t source) { source_ = std::make_optional(std::move(source)); }

    void pre_init(std::size_t faultNo, Vector<double>& state, LinearAllocator&) const;
    void init(std::size_t faultNo, Matrix<double> const& traction, Vector<double>& state,
              LinearAllocator&) const;

    double rhs(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double>& result, LinearAllocator&) const;
    void state(std::size_t faultNo, Matrix<double> const& traction, Vector<double const>& state,
               Matrix<double>& result, LinearAllocator&) const;

private:
    Law law_;
    std::optional<source_fun_t> source_;
};

template <class Law>
void RateAndState<Law>::pre_init(std::size_t faultNo, Vector<double>& state,
                                 LinearAllocator&) const {

    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        state(node) = law_.S_init(index + node);
    }
}

template <class Law>
void RateAndState<Law>::init(std::size_t faultNo, Matrix<double> const& traction,
                             Vector<double>& state, LinearAllocator&) const {

    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        state(nbf + node) = law_.psi_init(index + node, traction(node, 0));
    }
}

template <class Law>
double RateAndState<Law>::rhs(std::size_t faultNo, double time, Matrix<double> const& traction,
                              Vector<double const>& state, Vector<double>& result,
                              LinearAllocator&) const {
    double VMax = 0.0;
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto tau = traction(node, 0);
        auto psi = state(nbf + node);
        double V = law_.slip_rate(index + node, tau, psi);
        VMax = std::max(VMax, std::fabs(V));
        result(node) = V;
        result(nbf + node) = law_.state_rhs(index + node, V, psi);
    }
    if (source_) {
        auto coords = fault_[faultNo].template get<Coords>();
        std::array<double, DomainDimension + 1> xt;
        for (std::size_t node = 0; node < nbf; ++node) {
            auto const& x = coords[node];
            std::copy(x.begin(), x.end(), xt.begin());
            xt.back() = time;
            result(nbf + node) += (*source_)(xt)[0];
        }
    }
    return VMax;
}

template <class Law>
void RateAndState<Law>::state(std::size_t faultNo, Matrix<double> const& traction,
                              Vector<double const>& state, Matrix<double>& result,
                              LinearAllocator&) const {
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto tau = traction(node, 0);
        auto psi = state(nbf + node);
        double V = law_.slip_rate(index + node, tau, psi);
        result(node, 0) = psi;
        result(node, 1) = state(node);
        result(node, 2) = law_.tau_pre(index + node) + tau;
        result(node, 3) = V;
    }
}

} // namespace tndm

#endif // RATEANDSTATE_20201026_H
