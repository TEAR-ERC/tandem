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

    static constexpr double MIN_SN  = 10.0;
    static constexpr double MAX_SN  = 110.0;
    static constexpr double MIN_TAU = 25.0 * 0.6;  // = 15.0
    static constexpr double MAX_TAU = 70.0 * 0.6;  // = 42.0
    static constexpr double norm_init  = 56.588; //(50^2+26.5^2)^0.5
    static constexpr double normFactor  = 5;

class DieterichRuinaAgeing {
public:
    static constexpr std::size_t TangentialComponents = DomainDimension - 1u;



    struct ConstantParams {
        double V0;
        double b;
    
    };
    struct Params {
        double a;
        double eta;
        double L;
        double sn_pre;
        double base_fric;
        std::array<double, TangentialComponents> tau_pre;
        std::array<double, TangentialComponents> Vinit;
        std::array<double, TangentialComponents> Sinit;
    };



    static double clampByNormFactor(double& sn, double& tau) {
        double rawNorm = std::hypot(sn, tau);
        double rho     = rawNorm / norm_init;
    
        double scale = 1.0;
    
        if (rho > normFactor) {
            scale = normFactor / rho;  // scale down
        } else if (rho < 1.0 / normFactor) {
            scale = (1.0 / normFactor) / rho;  // scale up
        }
    
        return scale;
    }

    static double clampSnTauByRatio(double& snVal, double& tauVal) {
        double scale = 1.0;
        if      (snVal < MIN_SN)  scale = MIN_SN  / snVal;
        else if (snVal > MAX_SN)  scale = MAX_SN  / snVal;
        else if (tauVal < MIN_TAU) scale = MIN_TAU / tauVal;
        else if (tauVal > MAX_TAU) scale = MAX_TAU / tauVal;
    
        return scale;
    }

    static double verifyStress(double val, double minVal, double maxVal) {
        if (val < minVal) return minVal;
        if (val > maxVal) return maxVal;
        return val;
    }

    void set_num_nodes(std::size_t num_nodes) { p_.resize(num_nodes); }
    void set_constant_params(ConstantParams const& params) { cp_ = params; }
    void set_params(std::size_t index, Params const& params) {
        p_[index].get<A>() = params.a;
        p_[index].get<Eta>() = params.eta;
        p_[index].get<L>() = params.L;
        p_[index].get<base_fric>() = params.base_fric;
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
            return p_[index].get<base_fric>();
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

    auto slip_rate(std::size_t index, std::size_t fault_index, double sn,
        std::array<double, TangentialComponents> const& tau, double psi,
        std::array<double, DomainDimension> const& x, int* _ierr) const
-> std::array<double, TangentialComponents> {
auto eta = p_[index].get<Eta>();
auto tauAbsVec = tau + p_[index].get<TauPre>();
double snAbs = -sn + p_[index].get<SnPre>();
double tauAbs = norm(tauAbsVec);
double V = 0.0;
int ierr = 0;

//double scale = clampSnTauByRatio(snAbs, tauAbs);
//tauAbs *= scale;
// snAbs *= scale;
snAbs= verifyStress(snAbs, MIN_SN, MAX_SN);

if (eta == 0.0) {
 V = Finv(index, snAbs, tauAbs, psi);
} else {
 if (snAbs <= 0.0) { // fault is experiencing tension/opening
     snAbs = 0.0;
     V = tauAbs / eta;
     ierr = 1;
 } else {
     double Va = 0.0;
     double Vb = tauAbs / eta;
     if (Va > Vb) {
         std::swap(Va, Vb);
     }
     auto fF = [this, &index, &snAbs, &tauAbs, &psi, &eta](double V) {
         return tauAbs - this->F(index, snAbs, V, psi) - eta * V;
     };
     try {
         V = zeroIn(Va, Vb, fF);
     } catch (std::exception const&) {
         V = NAN;
         ierr = 2;
     }
     int rank;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     
     if (ierr != 0 ) {


        auto _A     = p_[index].get<A>();
        auto _Eta   = p_[index].get<Eta>();
        auto _L     = p_[index].get<L>();
        auto _SnPre = p_[index].get<SnPre>();
        auto f0     = p_[index].get<base_fric>();
        auto const& tau_pre = p_[index].get<TauPre>();

        std::cout << "rank [" << rank << "], fault_basis_index [" << index 
                  << "], fault_index [" << fault_index << "]\n"
                   << "  ierr = " << ierr << "\n"
                   << "  f0 = " << f0 << "\n"
                   << "  V0 = " << cp_.V0 << " (const)\n"
                   << "  b  = " << cp_.b << " (const)\n"
                   << "  a(x)   = " << _A << "\n"
                   << "  eta(x) = " << _Eta << "\n"
                   << "  L(x)   = " << _L << "\n"
                   << "  sigma_n_pre(x) = " << _SnPre << "\n"
                   << "  tau_pre(x)     = { ";
         for (std::size_t t = 0; t < DomainDimension - 2; ++t)
             std::cout << tau_pre[t] << ", ";
         std::cout << tau_pre[DomainDimension - 2] << " }\n"
                   << "  sigma_n = " << snAbs << "\n"
                   << "  |tau|   = " << tauAbs << "\n"
                   << "  psi     = " << psi << "\n"
                   << "  V_lower = " << Va << "\n"
                   << "  V_upper = " << Vb << "\n"
                   << "  R(V_lower) = " << fF(Va) << "\n"
                   << "  R(V_upper) = " << fF(Vb) << "\n"
                   << "  x = { ";
         for (std::size_t t = 0; t < DomainDimension - 1; ++t)
             std::cout << x[t] << ", ";
         std::cout << x[DomainDimension - 1] << " }\n";
     }
 }
}
*_ierr = ierr;
return -(V / tauAbs) * tauAbsVec;
}


    double state_rhs(std::size_t index, double V, double psi) const {
        double myL = p_[index].get<L>();
        double f0 = p_[index].get<base_fric>();
        return cp_.b * cp_.V0 / myL * (exp((f0 - psi) / cp_.b) - V / cp_.V0);
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
    struct base_fric {
        using type = double;
    };
    struct Vinit {
        using type = std::array<double, TangentialComponents>;
    };
    struct Sinit {
        using type = std::array<double, TangentialComponents>;
    };
    mneme::MultiStorage<mneme::DataLayout::SoA, SnPre, TauPre, A, Eta, L,base_fric, Vinit, Sinit> p_;
};

} // namespace tndm

#endif // DIETERICHRUINAAGEING_20201027_H
