#ifndef RATEANDSTATE_20201026_H
#define RATEANDSTATE_20201026_H

#include "config.h"
#include "tandem/RateAndStateBase.h"

#include "geometry/Vector.h"
#include "tensor/Reshape.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <optional>

namespace tndm {

template <class Law> class RateAndState : public RateAndStateBase {
public:
    using RateAndStateBase::RateAndStateBase;
    static constexpr std::size_t PsiIndex = TangentialComponents;

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

    void pre_init(std::size_t faultNo, Vector<double>& state, LinearAllocator<double>&) const;
    double init(std::size_t faultNo, Matrix<double> const& traction, Vector<double>& state,
                LinearAllocator<double>&) const;

    double rhs(std::size_t faultNo, double time, Matrix<double> const& traction,
               Vector<double const>& state, Vector<double>& result, LinearAllocator<double>&) const;

    auto state_prototype(std::size_t numLocalElements) const;
    void state(std::size_t faultNo, Matrix<double> const& traction, Vector<double const>& state,
               Matrix<double>& result, LinearAllocator<double>&) const;
    auto params_prototype(std::size_t numLocalElements) const;
    void params(std::size_t faultNo, Matrix<double>& result, LinearAllocator<double>&) const;

private:
    template <typename T> auto mat(Vector<T>& state) const {
        std::size_t nbf = space_.numBasisFunctions();
        return reshape(state, nbf, NumQuantities);
    }
    auto get_tau(std::size_t node, Matrix<double> const& traction) const {
        std::array<double, TangentialComponents> result;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            result[t] = traction(node, t + 1);
        }
        return result;
    }

    Law law_;
    std::optional<source_fun_t> source_;
};

template <class Law>
void RateAndState<Law>::pre_init(std::size_t faultNo, Vector<double>& state,
                                 LinearAllocator<double>&) const {
    auto s_mat = mat(state);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto Sini = law_.S_init(index + node);
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            s_mat(node, t) = Sini[t];
        }
    }
}

template <class Law>
double RateAndState<Law>::init(std::size_t faultNo, Matrix<double> const& traction,
                               Vector<double>& state, LinearAllocator<double>&) const {
    double VMax = 0.0;
    auto s_mat = mat(state);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = traction(node, 0);
        auto tau = get_tau(node, traction);
        auto psi = law_.psi_init(index + node, sn, tau);
        double V = norm(law_.slip_rate(index + node, sn, tau, psi));
        VMax = std::max(VMax, V);
        s_mat(node, PsiIndex) = psi;
    }
    return VMax;
}

template <class Law>
double RateAndState<Law>::rhs(std::size_t faultNo, double time, Matrix<double> const& traction,
                              Vector<double const>& state, Vector<double>& result,
                              LinearAllocator<double>&) const {
    double VMax = 0.0;
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    auto s_mat = mat(state);
    auto r_mat = mat(result);
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = traction(node, 0);
        auto psi = s_mat(node, PsiIndex);
        auto Vi = law_.slip_rate(index + node, sn, get_tau(node, traction), psi);
        double V = norm(Vi);
        VMax = std::max(VMax, V);
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            r_mat(node, t) = Vi[t];
        }
        r_mat(node, PsiIndex) = law_.state_rhs(index + node, V, psi);
    }
    if (source_) {
        auto coords = fault_[faultNo].template get<Coords>();
        std::array<double, DomainDimension + 1> xt;
        for (std::size_t node = 0; node < nbf; ++node) {
            auto const& x = coords[node];
            std::copy(x.begin(), x.end(), xt.begin());
            xt.back() = time;
            r_mat(node, PsiIndex) += (*source_)(xt)[0];
        }
    }
    return VMax;
}

template <class Law> auto RateAndState<Law>::state_prototype(std::size_t numLocalElements) const {
    auto names = std::vector<std::string>(2 + 3 * TangentialComponents);
    char buf[100];

    std::size_t out = 0;
    names[out++] = "state";
    for (std::size_t t = 0; t < TangentialComponents; ++t) {
        snprintf(buf, sizeof(buf), "slip%lu", t);
        names[out++] = buf;
    }
    for (std::size_t t = 0; t < TangentialComponents; ++t) {
        snprintf(buf, sizeof(buf), "traction%lu", t);
        names[out++] = buf;
    }
    for (std::size_t t = 0; t < TangentialComponents; ++t) {
        snprintf(buf, sizeof(buf), "slip-rate%lu", t);
        names[out++] = buf;
    }
    names[out++] = "normal-stress";
    return FiniteElementFunction<DomainDimension - 1u>(space_.clone(), names, numLocalElements);
}

template <class Law>
void RateAndState<Law>::state(std::size_t faultNo, Matrix<double> const& traction,
                              Vector<double const>& state, Matrix<double>& result,
                              LinearAllocator<double>&) const {
    auto s_mat = mat(state);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto sn = traction(node, 0);
        auto tau = get_tau(node, traction);
        auto psi = s_mat(node, PsiIndex);
        auto V = law_.slip_rate(index + node, sn, tau, psi);
        auto tau_hat = law_.tau_hat(index + node, tau, V);
        std::size_t out = 0;
        result(node, out++) = psi;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            result(node, out++) = s_mat(node, t);
        }
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            result(node, out++) = tau_hat[t];
        }
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            result(node, out++) = V[t];
        }
        result(node, out++) = law_.sn_hat(index + node, sn);
    }
}

template <class Law> auto RateAndState<Law>::params_prototype(std::size_t numLocalElements) const {
    return FiniteElementFunction<DomainDimension - 1u>(space_.clone(), law_.param_names(),
                                                       numLocalElements);
}

template <class Law>
void RateAndState<Law>::params(std::size_t faultNo, Matrix<double>& result,
                               LinearAllocator<double>&) const {

    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto row = result.subtensor(node, slice{});
        law_.params(index + node, row);
    }
}

} // namespace tndm

#endif // RATEANDSTATE_20201026_H
