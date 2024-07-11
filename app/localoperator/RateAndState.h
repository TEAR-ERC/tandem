#ifndef RATEANDSTATE_20201026_H
#define RATEANDSTATE_20201026_H

#include "RateAndStateBase.h"
#include "config.h"

#include "geometry/Vector.h"
#include "tensor/Reshape.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"
#include "../tandem/FrictionConfig.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <optional>
#include <iostream>
#include <string>

namespace tndm {

template <class Law> class RateAndState : public RateAndStateBase {
public:
    using RateAndStateBase::RateAndStateBase;
    static constexpr std::size_t PsiIndex = TangentialComponents;

    using param_fun_t =
        std::function<typename Law::Params(std::array<double, DomainDimension> const&)>;
    using source_fun_t =
        std::function<std::array<double, 1>(std::array<double, DomainDimension + 1> const&)>;
    using delta_tau_fun_t = std::function<std::array<double, TangentialComponents>(
        std::array<double, DomainDimension + 1> const&)>;
    using delta_sn_fun_t =
        std::function<std::array<double, 1>(std::array<double, DomainDimension + 1> const&)>;

    void set_constant_params(typename Law::ConstantParams const& cps) {
        law_.set_constant_params(cps);
    }



    void set_params(param_fun_t pfun) {
        std::string newfile="newfault.lua";
        std::string newscanerio="fault1";
        
        DieterichRuinaAgeingScenario readnewfault(newfile, newscanerio);

        auto num_nodes = fault_.storage().size();
        law_.set_num_nodes(num_nodes);
        for (std::size_t index = 0; index < num_nodes; ++index) {
            auto params = pfun(fault_.storage()[index].template get<Coords>());
            law_.set_params(index, params);
        }

        auto param_generator = readnewfault.param_fun();
        auto params1= param_generator(fault_.storage()[0].template get<Coords>());
        law_.set_params(0, params1);

    }

    void set_source_fun(source_fun_t source) { source_ = std::make_optional(std::move(source)); }
    void set_delta_tau_fun(delta_tau_fun_t delta_tau) {
        delta_tau_ = std::make_optional(std::move(delta_tau));
    }
    void set_delta_sn_fun(delta_sn_fun_t delta_sn) {
        delta_sn_ = std::make_optional(std::move(delta_sn));
    }

    void pre_init(std::size_t faultNo, Vector<double>& state, LinearAllocator<double>&) const;
    double init(double time, std::size_t faultNo, Vector<double const> const& traction,
                Vector<double>& state, LinearAllocator<double>&, int* ierr) const;

    double rhs(double time, std::size_t faultNo, Vector<double const> const& traction,
               Vector<double const>& state, Vector<double>& result, LinearAllocator<double>&,
               int* ierr) const;

    auto state_prototype(std::size_t numLocalElements) const;
    void state(double time, std::size_t faultNo, Vector<double const> const& traction,
               Vector<double const>& state, Matrix<double>& result, LinearAllocator<double>&) const;
    auto params_prototype(std::size_t numLocalElements) const;
    void params(std::size_t faultNo, Matrix<double>& result, LinearAllocator<double>&) const;

private:
    template <typename T> auto state_mat(Vector<T>& state) const {
        std::size_t nbf = space_.numBasisFunctions();
        return reshape(state, nbf, NumQuantities);
    }
    template <typename T> auto traction_mat(Vector<T> const& traction) const {
        std::size_t nbf = space_.numBasisFunctions();
        return reshape(traction, nbf, DomainDimension);
    }
    template <typename T> auto get_tau(std::size_t node, Matrix<T> const& traction) const {
        std::array<double, TangentialComponents> result;
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            result[t] = traction(node, t + 1);
        }
        return result;
    }
    template <typename T> auto get_sn(std::size_t node, Matrix<T> const& traction) const {
        std::array<double, 1> result;
        return result;
    }
    auto get_delta_tau(double time, std::size_t faultNo, std::size_t node) const {
        auto x = fault_[faultNo].template get<Coords>()[node];
        std::array<double, DomainDimension + 1> xt;
        std::copy(x.begin(), x.end(), xt.begin());
        xt.back() = time;
        return (*delta_tau_)(xt);
    }
    auto get_delta_sn(double time, std::size_t faultNo, std::size_t node) const {
        auto x = fault_[faultNo].template get<Coords>()[node];
        std::array<double, DomainDimension + 1> xt;
        std::copy(x.begin(), x.end(), xt.begin());
        xt.back() = time;
        return (*delta_sn_)(xt)[0];
    }

    Law law_;
    std::optional<source_fun_t> source_;
    std::optional<delta_tau_fun_t> delta_tau_;
    std::optional<delta_sn_fun_t> delta_sn_;
};

template <class Law>
void RateAndState<Law>::pre_init(std::size_t faultNo, Vector<double>& state,
                                 LinearAllocator<double>&) const {
    auto s_mat = state_mat(state);
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
double RateAndState<Law>::init(double time, std::size_t faultNo,
                               Vector<double const> const& traction, Vector<double>& state,
                               LinearAllocator<double>&, int* _ierr) const {
    double VMax = 0.0;
    auto s_mat = state_mat(state);
    auto t_mat = traction_mat(traction);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    auto coords = fault_[faultNo].template get<Coords>();
    int ierr = 0, ierr_max = 0;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto const& x = coords[node];
        auto sn = t_mat(node, 0);
        if (delta_sn_) {
            sn = sn + get_delta_sn(time, faultNo, node);
        }
        auto tau = get_tau(node, t_mat);
        if (delta_tau_) {
            tau = tau + get_delta_tau(time, faultNo, node);
        }
        auto psi = law_.psi_init(index + node, sn, tau);
        double V = norm(law_.slip_rate(index + node, faultNo, sn, tau, psi, x, &ierr));
        if (ierr > ierr_max) {
            ierr_max = ierr;
        }
        VMax = std::max(VMax, V);
        s_mat(node, PsiIndex) = psi;
    }
    *_ierr = ierr_max;
    return VMax;
}

template <class Law>
double RateAndState<Law>::rhs(double time, std::size_t faultNo,
                              Vector<double const> const& traction, Vector<double const>& state,
                              Vector<double>& result, LinearAllocator<double>&, int* _ierr) const {
    double VMax = 0.0;
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    auto s_mat = state_mat(state);
    auto r_mat = state_mat(result);
    auto t_mat = traction_mat(traction);
    auto coords = fault_[faultNo].template get<Coords>();
    int ierr = 0, ierr_max = 0;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto const& x = coords[node];

        auto sn = t_mat(node, 0);
        if (delta_sn_) {
            sn = sn + get_delta_sn(time, faultNo, node);
        }
        auto psi = s_mat(node, PsiIndex);
        auto tau = get_tau(node, t_mat);
        if (delta_tau_) {
            tau = tau + get_delta_tau(time, faultNo, node);
        }

        auto Vi = law_.slip_rate(index + node, faultNo, sn, tau, psi, x, &ierr);
        if (ierr > ierr_max) {
            ierr_max = ierr;
        }

        double V = norm(Vi);
        VMax = std::max(VMax, V);
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            r_mat(node, t) = Vi[t];
        }
        r_mat(node, PsiIndex) = law_.state_rhs(index + node, V, psi);
    }
    if (source_) {
        std::array<double, DomainDimension + 1> xt;
        for (std::size_t node = 0; node < nbf; ++node) {
            auto const& x = coords[node];
            std::copy(x.begin(), x.end(), xt.begin());
            xt.back() = time;
            r_mat(node, PsiIndex) += (*source_)(xt)[0];
        }
    }
    *_ierr = ierr_max;
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
void RateAndState<Law>::state(double time, std::size_t faultNo,
                              Vector<double const> const& traction, Vector<double const>& state,
                              Matrix<double>& result, LinearAllocator<double>&) const {
    auto s_mat = state_mat(state);
    auto t_mat = traction_mat(traction);
    std::size_t nbf = space_.numBasisFunctions();
    std::size_t index = faultNo * nbf;
    auto coords = fault_[faultNo].template get<Coords>();
    int ierr;
    for (std::size_t node = 0; node < nbf; ++node) {
        auto const& x = coords[node];
        auto sn = t_mat(node, 0);
        if (delta_sn_) {
            sn = sn + get_delta_sn(time, faultNo, node);
        }
        auto tau = get_tau(node, t_mat);
        if (delta_tau_) {
            tau = tau + get_delta_tau(time, faultNo, node);
        }
        auto psi = s_mat(node, PsiIndex);
        auto V = law_.slip_rate(index + node, faultNo, sn, tau, psi, x, &ierr);
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
