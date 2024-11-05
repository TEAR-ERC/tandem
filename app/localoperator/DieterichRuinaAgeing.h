#ifndef DIETERICHRUINAAGEING_20201027_H
#define DIETERICHRUINAAGEING_20201027_H

#include "config.h"

#include "geometry/Vector.h"
#include "tensor/Tensor.h"
#include "util/Zero.h"

#include "mneme/storage.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <iostream>

namespace tndm {

class DieterichRuinaAgeing {
public:
    static constexpr std::size_t TangentialComponents = DomainDimension - 1u;

    struct ConstantParams {
        double V0;
        double b;
        double f0;
    };
    struct Params {
        double a;
        double eta;
        double L;
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
        p_[index].get<L>() = params.L;
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
        if (Vi == 0.0) {
            return cp_.f0;
        }
        auto a = p_[index].get<A>();
        auto eta = p_[index].get<Eta>();
        double s = sinh((tauAbs - eta * Vi) / (a * snAbs));
        double l = log((2.0 * cp_.V0 / Vi) * s);
        return a * l;
    }

    /**
     * @brief Absolute normal stress on fault (positive in compression)
     */
    double sn_hat(std::size_t index, double sn) const { return -sn + p_[index].get<SnPre>(); }
    /**
     * @brief Absolute shear stress on fault.
     *
     * Includes radiation damping term. I.e. the returned value is equivalent to -sn_hat f(V, psi).
     */
    auto tau_hat(std::size_t index, std::array<double, TangentialComponents> const& tau,
                 std::array<double, TangentialComponents> const& V) const {
        return tau + p_[index].get<TauPre>() + p_[index].get<Eta>() * V;
    }
    auto S_init(std::size_t index) const { return p_[index].get<Sinit>(); }

    auto slip_rate(std::size_t index, double sn,
                   std::array<double, TangentialComponents> const& tau, double psi) const
        -> std::array<double, TangentialComponents> {
        auto eta = p_[index].get<Eta>();
        auto tauAbsVec = tau + p_[index].get<TauPre>();
        double snAbs = -sn + p_[index].get<SnPre>();
        double tauAbs = norm(tauAbsVec);
        double V = 0.0;
        double a = -32;
        double a_min = std::log10(std::nextafter(0, 1));
        double b;
        if (eta == 0.0) {
            V = Finv(index, snAbs, tauAbs, psi);
        } else {
            if (snAbs <= 0.0) { /* Implies the fault is experiencing tension / opening */
                snAbs = 0.0;    /* Just to illustrate what we are doing */
                /* Solve R(V) = T - sigma_n F(V,psi) - eta V with sigma_n = 0.0 */
                V = tauAbs / eta;
            } else {
                b = std::log10(tauAbs / eta);
                auto fF = [this, &index, &snAbs, &tauAbs, &psi, &eta](double Ve) {
                    return tauAbs - this->F(index, snAbs, std::pow(10.0, Ve), psi) -
                           eta * std::pow(10.0, Ve);
                };
                try {
                    auto Ve = zeroIn(a, b, fF);
                    V = std::pow(10.0, Ve);
                } catch (std::exception const&) {
                    try {
                        auto Ve = zeroIn(a_min, b, fF);
                        V = std::pow(10.0, Ve);
                    } catch (std::exception const&) {
                        std::cout << "sigma_n = " << snAbs << std::endl
                                  << "|tau| = " << tauAbs << std::endl
                                  << "psi = " << psi << std::endl
                                  << "L = " << a << std::endl
                                  << "U = " << b << std::endl
                                  << "F(L) = " << fF(a) << std::endl
                                  << "F(U) = " << fF(b) << std::endl;
                        throw;
                    }
                }
            }
        }
        return -(V / tauAbs) * tauAbsVec;
    }

    double state_rhs(std::size_t index, double V, double psi) const {
        double myL = p_[index].get<L>();
        return cp_.b * cp_.V0 / myL * (exp((cp_.f0 - psi) / cp_.b) - V / cp_.V0);
    }

    auto param_names() const {
        auto names = std::vector<std::string>(4 + TangentialComponents);
        char buf[100];

        std::size_t i = 0;
        names[i++] = "a";
        names[i++] = "eta";
        names[i++] = "L";
        names[i++] = "sn_pre";
        for (std::size_t t = 0; t < TangentialComponents; ++t) {
            snprintf(buf, sizeof(buf), "tau_pre%lu", t);
            names[i++] = buf;
        }
        return names;
    }

    template <typename VecT> void params(std::size_t index, VecT& result) const {
        assert(result.shape(0) == 4 + TangentialComponents);
        std::ptrdiff_t i = 0;
        result(i++) = p_[index].get<A>();
        result(i++) = p_[index].get<Eta>();
        result(i++) = p_[index].get<L>();
        result(i++) = p_[index].get<SnPre>();
        auto const& tau_pre = p_[index].get<TauPre>();
        for (auto const& t : tau_pre) {
            result(i++) = t;
        }
    }

private:
    double F(std::size_t index, double snAbs, double V, double psi) const {
        auto a = p_[index].get<A>();
        double e = exp(psi / a);
        double f = a * asinh((V / (2.0 * cp_.V0)) * e);
        return snAbs * f;
    }

    double Finv(std::size_t index, double snAbs, double tauAbs, double psi) const {
        // We have
        // V = 2 V_0 sinh(tauAbs / (a snAbs)) exp(-psi / a)
        // substituting r = tauAbs / snAbs and sinh(x) = (exp(x) - exp(-x)) / 2 we obtain
        // V = V_0 (exp((r - psi) / a) - exp(-(r + psi) / a))
        auto a = p_[index].get<A>();
        double r = tauAbs / snAbs;
        return cp_.V0 * (exp((r - psi) / a) - exp(-(r + psi) / a));
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
    struct L {
        using type = double;
    };
    struct Vinit {
        using type = std::array<double, TangentialComponents>;
    };
    struct Sinit {
        using type = std::array<double, TangentialComponents>;
    };
    mneme::MultiStorage<mneme::DataLayout::SoA, SnPre, TauPre, A, Eta, L, Vinit, Sinit> p_;
};

} // namespace tndm

#endif // DIETERICHRUINAAGEING_20201027_H
