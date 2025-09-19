#ifndef ELASTICITYSCENARIO_20200930_H
#define ELASTICITYSCENARIO_20200930_H

#include "common/Scenario.h"
#include "config.h"
#include "form/DGCurvilinearCommon.h"
#include "localoperator/Elasticity.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace tndm {

class ElasticityScenario : public Scenario<Elasticity> {
public:
    constexpr static char Mu[] = "mu";
    constexpr static char Mu0[] = "mu0";
    constexpr static char Theta[] = "theta";
    constexpr static char Mu1[] = "mu1";
    constexpr static char Viscosity[] = "viscosity";
    constexpr static char RelaxationTime[] = "relaxationTime";
    constexpr static char Lam[] = "lam";

    ElasticityScenario(std::string const& lib, std::string const& scenario,
                       std::array<double, DomainDimension> const& ref_normal)
        : Scenario(lib, scenario, ref_normal) {
        if (lib_.hasMember(scenario, Mu)) {
            mu_ = lib_.getMemberFunction<DomainDimension, 1>(scenario, Mu);
        }
        if (lib_.hasMember(scenario, Lam)) {
            lam_ = lib_.getMemberFunction<DomainDimension, 1>(scenario, Lam);
        }
        if (lib_.hasMember(scenario, Mu0)) {
            mu0_ = lib_.getMemberFunction<DomainDimension, 1>(scenario, Mu0);
        }
        if (lib_.hasMember(scenario, Mu1)) {
            mu1_ = lib_.getMemberFunction<DomainDimension, 1>(scenario, Mu1);
        }
        if (lib_.hasMember(scenario, Viscosity)) {
            viscosity_ = lib_.getMemberFunction<DomainDimension, 1>(scenario, Viscosity);
        }
        if (lib_.hasMember(scenario, Theta)) {
            theta_ = lib_.getMemberConstant(scenario, Theta);
        }
        if (lib_.hasMember(scenario, RelaxationTime)) {
            relaxationTime_ = lib_.getMemberFunction<DomainDimension, 1>(scenario, RelaxationTime);
        }
    }

    auto const& lam() const { return lam_; }
    auto const& mu() const { return mu_; }
    auto const& mu0() const { return mu0_; }
    auto const& mu1() const { return mu1_; }
    auto const& viscosity() const { return viscosity_; }
    auto const& theta() const { return theta_; }
    auto const& relaxation_time() const { return relaxationTime_; }

    auto make_local_operator(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                             DGMethod method) const {
        auto elasticity =
            std::make_shared<Elasticity>(std::move(cl), lam_, mu_, mu0_, mu1_, viscosity_,
                                         relaxationTime_, theta_, std::nullopt, method);
        set(*elasticity);
        return elasticity;
    }

private:
    functional_t<1> lam_ =
        [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> { return {1.0}; };
    functional_t<1> mu_ =
        [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> { return {1.0}; };
    functional_t<1> mu0_ =
        [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> { return {1.0}; };
    functional_t<1> mu1_ =
        [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> { return {1.0}; };
    functional_t<1> viscosity_ =
        [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> { return {1.0}; };
    functional_t<1> relaxationTime_ =
        [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> { return {1.0}; };
    double theta_ = 0.0;
};

} // namespace tndm

#endif // ELASTICITYSCENARIO_20200930_H
