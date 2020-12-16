#ifndef DIETERICHRUINAAGEING_20201027_H
#define DIETERICHRUINAAGEING_20201027_H

#include "RateAndState.h"

#include "geometry/Vector.h"
#include "util/Zero.h"

#include <algorithm>
#include <cassert>

namespace tndm {

class DieterichRuinaAgeing {
public:
    static constexpr std::size_t TangentialComponents = DomainDimension - 1u;

    struct ConstantParams {
        double V0;
        double b;
        double L;
        double f0;
    };
    struct Params {
        double a;
        double eta;
        double sn_pre;
        std::array<double, TangentialComponents> tau_pre;
        std::array<double, TangentialComponents> Vinit;
        std::array<double, TangentialComponents> Sinit;
    };

    void set_num_nodes(std::size_t num_nodes) { p_.resize(num_nodes); }
    void set_constant_params(ConstantParams const& params) { cp_ = params; }
    void set_params(std::size_t index, Params const& params) {
        p_[index].get<A>() = params.a;
        p_[index].get<Eta>() = params.eta;
        p_[index].get<SnPre>() = params.sn_pre;
        p_[index].get<TauPre>() = params.tau_pre;
        p_[index].get<Vinit>() = params.Vinit;
        p_[index].get<Sinit>() = params.Sinit;
    }

    double psi_init(std::size_t index, double sn,
                    std::array<double, TangentialComponents> const& tau) const {
        double snAbs = -sn + p_[index].get<SnPre>();
        double tauAbs = norm(tau + p_[index].get<TauPre>());
        auto Vi = norm(p_[index].get<Vinit>());
        auto a = p_[index].get<A>();
        auto eta = p_[index].get<Eta>();
        double s = sinh((tauAbs - eta * Vi) / (a * snAbs));
        double l = log((2.0 * cp_.V0 / Vi) * s);
        return a * l;
    }

    double sn_pre(std::size_t index) const { return p_[index].get<SnPre>(); }
    auto tau_pre(std::size_t index) const { return p_[index].get<TauPre>(); }
    auto S_init(std::size_t index) const { return p_[index].get<Sinit>(); }

    auto slip_rate(std::size_t index, double sn,
                   std::array<double, TangentialComponents> const& tau, double psi) const
        -> std::array<double, TangentialComponents> {
        auto eta = p_[index].get<Eta>();
        auto tauAbsVec = tau + p_[index].get<TauPre>();
        double snAbs = -sn + p_[index].get<SnPre>();
        double tauAbs = norm(tauAbsVec);
        double a = 0.0;
        double b = tauAbs / eta;
        if (a > b) {
            std::swap(a, b);
        }
        auto fF = [this, &index, &snAbs, &tauAbs, &psi, &eta](double V) {
            return tauAbs - this->F(index, snAbs, V, psi) - eta * V;
        };
        double V = zeroIn(a, b, fF);
        return (V / (F(index, snAbs, V, psi) + eta * V)) * tauAbsVec;
    }

    double state_rhs(std::size_t index, double V, double psi) const {
        return cp_.b * cp_.V0 / cp_.L * (exp((cp_.f0 - psi) / cp_.b) - V / cp_.V0);
    }

private:
    double F(std::size_t index, double sn, double V, double psi) const {
        auto a = p_[index].get<A>();
        double e = exp(psi / a);
        double f = a * asinh((V / (2.0 * cp_.V0)) * e);
        return sn * f;
    }

    ConstantParams cp_;

    struct SnPre {
        using type = double;
    };
    struct TauPre {
        using type = std::array<double, TangentialComponents>;
    };
    struct A {
        using type = double;
    };
    struct Eta {
        using type = double;
    };
    struct Vinit {
        using type = std::array<double, TangentialComponents>;
    };
    struct Sinit {
        using type = std::array<double, TangentialComponents>;
    };
    mneme::MultiStorage<mneme::DataLayout::SoA, SnPre, TauPre, A, Eta, Vinit, Sinit> p_;
};

} // namespace tndm

#endif // DIETERICHRUINAAGEING_20201027_H
