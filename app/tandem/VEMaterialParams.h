#ifndef VEMATERIALPARAMS_20260328_H
#define VEMATERIALPARAMS_20260328_H

#include "config.h"
#include "script/LuaLib.h"

#include <array>
#include <optional>
#include <stdexcept>
#include <string>

namespace tndm {

template <class LocalOperator> struct VEMaterialParams {
    void load(LuaLib&, std::string const&) {}
};

class Viscoelasticity;

template <> struct VEMaterialParams<Viscoelasticity> {
    using functional_t_region = LuaLib::functional_t_region<DomainDimension, 1>;

    static constexpr char Mu0[] = "mu0";
    static constexpr char Mu1[] = "mu1";
    static constexpr char Viscosity[] = "viscosity";
    static constexpr char RelaxationTime[] = "relaxation_time";
    static constexpr char Theta[] = "theta";
    static constexpr char GTol[] = "g_tol";
    static constexpr char AllowNonpositiveTau[] = "allow_nonpositive_projected_relaxation_time";

    void load(LuaLib& lib, std::string const& scenario) {
        if (lib.hasMember(scenario, Mu0)) {
            mu0_ = lib.getMemberFunctionTagged<DomainDimension, 1>(scenario, Mu0);
        }
        if (lib.hasMember(scenario, Mu1)) {
            mu1_ = lib.getMemberFunctionTagged<DomainDimension, 1>(scenario, Mu1);
        }
        if (lib.hasMember(scenario, Viscosity)) {
            viscosity_ = lib.getMemberFunctionTagged<DomainDimension, 1>(scenario, Viscosity);
        }
        // Deprecated: tau is derived as viscosity / mu1. If given, relaxation_time is only
        // checked against it (Viscoelasticity::local_relaxation_time).
        if (lib.hasMember(scenario, RelaxationTime)) {
            relaxationTime_ =
                lib.getMemberFunctionTagged<DomainDimension, 1>(scenario, RelaxationTime);
        }

        if (!lib.hasMember(scenario, Theta)) {
            throw std::runtime_error("Missing '" + std::string(Theta) + "' in " + scenario);
        }
        theta_ = lib.getMemberConstant(scenario, Theta);
        if (theta_ <= 0.0) {
            throw std::runtime_error("Invalid '" + std::string(Theta) + "' value in " + scenario);
        }

        // Reassembly threshold on g(dt); see DGOperator::update_time_step. Bounds the
        // relative perturbation of the effective moduli A(dt), B(dt) by ~g_tol/g(theta).
        // Set to 0 to fall back to the legacy gate on the relative change in dt.
        if (lib.hasMember(scenario, GTol)) {
            gTol_ = lib.getMemberConstant(scenario, GTol);
            if (gTol_ < 0.0) {
                throw std::runtime_error("Invalid '" + std::string(GTol) + "' value in " +
                                         scenario);
            }
        }
        // Compatibility mode: run with a projected material that has viscosity / mu1 <= 0
        // somewhere (see DGOperator::check_projected_relaxation_times). Off by default.
        if (lib.hasMember(scenario, AllowNonpositiveTau)) {
            allowNonpositiveTau_ = lib.getMemberBoolean(scenario, AllowNonpositiveTau);
        }
    }

    auto const& mu0() const { return mu0_; }
    auto const& mu1() const { return mu1_; }
    auto const& viscosity() const { return viscosity_; }
    auto const& relaxation_time() const { return relaxationTime_; }
    double theta() const { return theta_; }
    double g_tol() const { return gTol_; }
    bool allow_nonpositive_projected_relaxation_time() const { return allowNonpositiveTau_; }

private:
    static functional_t_region constant_functional(double v) {
        return [v](std::array<double, DomainDimension> const&, long int) -> std::array<double, 1> {
            return {v};
        };
    }

    functional_t_region mu0_ = constant_functional(1.0);
    functional_t_region mu1_ = constant_functional(1.0);
    functional_t_region viscosity_ = constant_functional(1.0);
    std::optional<functional_t_region> relaxationTime_ = std::nullopt;
    double theta_ = 0.0;
    double gTol_ = 1.0e-8;
    bool allowNonpositiveTau_ = false;
};

class PoissonViscoelasticity;

// Unlike VEMaterialParams<Viscoelasticity>, there is no separate "relaxation_time"
// field: PoissonViscoelasticity derives tau = eta / mu1 per material node (see
// PoissonViscoelasticity.h). mu0 has its own dedicated "mu0" Lua field here,
// same as VEMaterialParams<Viscoelasticity>::Mu0 -- it must NOT be read via
// SeasScenario's generic mu() (that reads a "mu" field, not "mu0").
template <> struct VEMaterialParams<PoissonViscoelasticity> {
    using functional_t_region = LuaLib::functional_t_region<DomainDimension, 1>;

    static constexpr char Mu0[] = "mu0";
    static constexpr char Mu1[] = "mu1";
    static constexpr char Viscosity[] = "viscosity";
    static constexpr char Theta[] = "theta";
    static constexpr char AllowNonpositiveTau[] = "allow_nonpositive_projected_relaxation_time";

    void load(LuaLib& lib, std::string const& scenario) {
        if (lib.hasMember(scenario, Mu0)) {
            mu0_ = lib.getMemberFunctionTagged<DomainDimension, 1>(scenario, Mu0);
        }
        if (lib.hasMember(scenario, Mu1)) {
            mu1_ = lib.getMemberFunctionTagged<DomainDimension, 1>(scenario, Mu1);
        }
        if (lib.hasMember(scenario, Viscosity)) {
            viscosity_ = lib.getMemberFunctionTagged<DomainDimension, 1>(scenario, Viscosity);
        }
        if (lib.hasMember(scenario, Theta)) {
            theta_ = lib.getMemberConstant(scenario, Theta);
            if (theta_ <= 0.0) {
                throw std::runtime_error("Invalid '" + std::string(Theta) + "' value in " +
                                         scenario);
            }
        }
        // See VEMaterialParams<Viscoelasticity>.
        if (lib.hasMember(scenario, AllowNonpositiveTau)) {
            allowNonpositiveTau_ = lib.getMemberBoolean(scenario, AllowNonpositiveTau);
        }
    }

    auto const& mu0() const { return mu0_; }
    auto const& mu1() const { return mu1_; }
    auto const& viscosity() const { return viscosity_; }
    double theta() const { return theta_; }
    bool allow_nonpositive_projected_relaxation_time() const { return allowNonpositiveTau_; }

private:
    static functional_t_region constant_functional(double v) {
        return [v](std::array<double, DomainDimension> const&, long int) -> std::array<double, 1> {
            return {v};
        };
    }

    functional_t_region mu0_ = constant_functional(1.0);
    functional_t_region mu1_ = constant_functional(1.0);
    functional_t_region viscosity_ = constant_functional(1.0);
    double theta_ = 0.1;
    bool allowNonpositiveTau_ = false;
};

} // namespace tndm

#endif // VEMATERIALPARAMS_20260328_H
