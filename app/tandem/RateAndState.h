#ifndef RATEANDSTATE_20201026_H
#define RATEANDSTATE_20201026_H

#include "config.h"
#include "tandem/RateAndStateBase.h"

#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <array>
#include <cstddef>
#include <functional>

namespace tndm {

template <class Law> class RateAndState : public RateAndStateBase {
public:
    using RateAndStateBase::RateAndStateBase;

    using param_fun_t =
        std::function<typename Law::Params(std::array<double, DomainDimension> const&)>;

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

    void initial(std::size_t faultNo, Vector<double>& state, LinearAllocator& scratch) const;
    double rhs(std::size_t faultNo, Matrix<double> const& grad_u, Vector<double const>& state,
               Vector<double>& result, LinearAllocator& scratch) const;
    void state(std::size_t faultNo, Matrix<double> const& grad_u, Vector<double const>& state,
               Matrix<double>& result, LinearAllocator& scratch) const;

private:
    Law law_;
};

template <class Law>
void RateAndState<Law>::initial(std::size_t faultNo, Vector<double>& state,
                                LinearAllocator& scratch) const {

    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        state(node) = law_.psi0(index + node);
        state(nbf + node) = 0.0;
    }
}

template <class Law>
double RateAndState<Law>::rhs(std::size_t faultNo, Matrix<double> const& grad_u,
                              Vector<double const>& state, Vector<double>& result,
                              LinearAllocator& scratch) const {
    std::size_t nbf = space_.numBasisFunctions();
    double* traction_raw = scratch.allocate<double>(nbf);
    auto traction = Vector<double>(traction_raw, nbf);
    compute_traction(faultNo, grad_u, traction);

    double VMax = 0.0;
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto tau = traction(node);
        auto psi = state(node);
        double V = law_.slip_rate(index + node, tau, psi);
        VMax = std::max(VMax, std::fabs(V));
        result(node) = law_.state_rhs(index + node, V, psi);
        result(nbf + node) = V;
    }
    scratch.reset();
    return VMax;
}

template <class Law>
void RateAndState<Law>::state(std::size_t faultNo, Matrix<double> const& grad_u,
                              Vector<double const>& state, Matrix<double>& result,
                              LinearAllocator& scratch) const {
    std::size_t nbf = space_.numBasisFunctions();
    double* traction_raw = scratch.allocate<double>(nbf);
    auto traction = Vector<double>(traction_raw, nbf);
    compute_traction(faultNo, grad_u, traction);

    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto tau = traction(node);
        auto psi = state(node);
        double V = law_.slip_rate(index + node, tau, psi);
        result(node, 0) = psi;
        result(node, 1) = state(node + nbf);
        result(node, 2) = tau;
        result(node, 3) = V;
    }
    scratch.reset();
}

} // namespace tndm

#endif // RATEANDSTATE_20201026_H
