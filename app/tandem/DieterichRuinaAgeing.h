#ifndef DIETERICHRUINAAGEING_20201027_H
#define DIETERICHRUINAAGEING_20201027_H

#include "RateAndState.h"

#include "util/Zero.h"

#include <algorithm>
#include <cassert>

namespace tndm {

class DieterichRuinaAgeing {
public:
    struct ConstantParams {
        double eta;
        double Vinit;
        double sn;
        double V0;
        double b;
        double L;
        double f0;
    };
    struct Params {
        double a;
        double tau0;
    };

    void set_num_nodes(std::size_t num_nodes) { p_.resize(num_nodes); }
    void set_constant_params(ConstantParams const& params) { cp_ = params; }
    void set_params(std::size_t index, Params const& params) {
        p_[index].get<A>() = params.a;
        p_[index].get<Tau0>() = params.tau0;
    }

    double psi0(std::size_t index) const {
        auto tau0 = p_[index].get<Tau0>();
        auto a = p_[index].get<A>();
        double s = sinh((tau0 - cp_.eta * cp_.Vinit) / (a * cp_.sn));
        double l = log((2.0 * cp_.V0 / cp_.Vinit) * s);
        return a * l;
    }

    double slip_rate(std::size_t index, double tau, double psi) const {
        double tauAbs = tau + p_[index].get<Tau0>();
        double a = -tauAbs / cp_.eta;
        double b = tauAbs / cp_.eta;
        if (a > b) {
            std::swap(a, b);
        }
        auto fF = [this, &index, &tau, &psi](double V) { return this->F(index, tau, V, psi); };
        return zeroIn(a, b, fF);
    }

    double state_rhs(std::size_t index, double V, double psi) const {
        return cp_.b * cp_.V0 / cp_.L * (exp((cp_.f0 - psi) / cp_.b) - V / cp_.V0);
    }

private:
    double F(std::size_t index, double tau, double V, double psi) const {
        auto tau0 = p_[index].get<Tau0>();
        auto a = p_[index].get<A>();
        double e = exp(psi / a);
        double f = a * asinh((V / (2.0 * cp_.V0)) * e);
        return tau + tau0 - cp_.sn * f - cp_.eta * V;
    }

    ConstantParams cp_;

    struct Tau0 {
        using type = double;
    };
    struct A {
        using type = double;
    };
    mneme::MultiStorage<mneme::DataLayout::SoA, Tau0, A> p_;
};

} // namespace tndm

#endif // DIETERICHRUINAAGEING_20201027_H
