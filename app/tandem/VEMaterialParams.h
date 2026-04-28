#ifndef VEMATERIALPARAMS_20260328_H
#define VEMATERIALPARAMS_20260328_H

#include "config.h"
#include "script/LuaLib.h"

#include <array>
#include <stdexcept>
#include <string>

namespace tndm {

template <class LocalOperator> struct VEMaterialParams {
    void load(LuaLib&, std::string const&) {}
};

class Viscoelasticity;

template <> struct VEMaterialParams<Viscoelasticity> {
    using functional_t = LuaLib::functional_t<DomainDimension, 1>;

    static constexpr char Mu0[] = "mu0";
    static constexpr char Mu1[] = "mu1";
    static constexpr char Viscosity[] = "viscosity";
    static constexpr char RelaxationTime[] = "relaxation_time";
    static constexpr char Theta[] = "theta";

    void load(LuaLib& lib, std::string const& scenario) {
        if (lib.hasMember(scenario, Mu0)) {
            mu0_ = lib.getMemberFunction<DomainDimension, 1>(scenario, Mu0);
        }
        if (lib.hasMember(scenario, Mu1)) {
            mu1_ = lib.getMemberFunction<DomainDimension, 1>(scenario, Mu1);
        }
        if (lib.hasMember(scenario, Viscosity)) {
            viscosity_ = lib.getMemberFunction<DomainDimension, 1>(scenario, Viscosity);
        }
        if (lib.hasMember(scenario, RelaxationTime)) {
            relaxationTime_ = lib.getMemberFunction<DomainDimension, 1>(scenario, RelaxationTime);
        }

        if (!lib.hasMember(scenario, Theta)) {
            throw std::runtime_error("Missing '" + std::string(Theta) + "' in " + scenario);
        }
        theta_ = lib.getMemberConstant(scenario, Theta);
        if (theta_ <= 0.0) {
            throw std::runtime_error("Invalid '" + std::string(Theta) + "' value in " + scenario);
        }
    }

    auto const& mu0() const { return mu0_; }
    auto const& mu1() const { return mu1_; }
    auto const& viscosity() const { return viscosity_; }
    auto const& relaxation_time() const { return relaxationTime_; }
    double theta() const { return theta_; }

private:
    static functional_t constant_functional(double v) {
        return [v](std::array<double, DomainDimension> const&) -> std::array<double, 1> {
            return {v};
        };
    }

    functional_t mu0_ = constant_functional(1.0);
    functional_t mu1_ = constant_functional(1.0);
    functional_t viscosity_ = constant_functional(1.0);
    functional_t relaxationTime_ = constant_functional(1.0);
    double theta_ = 0.0;
};

} // namespace tndm

#endif // VEMATERIALPARAMS_20260328_H
