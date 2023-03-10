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
    constexpr static char Lam[] = "lam";

    ElasticityScenario(LocalSimplexMesh<DomainDimension> const& mesh, std::string const& lib, std::string const& scenario,
                       std::array<double, DomainDimension> const& ref_normal)
        : Scenario(mesh, lib, scenario, ref_normal) {
        if (lib_.hasMember(scenario, Mu)) {
            mu_ = lib_.getMemberFunction<DomainDimension+1, 1>(scenario, Mu);
        }
        if (lib_.hasMember(scenario, Lam)) {
            lam_ = lib_.getMemberFunction<DomainDimension+1, 1>(scenario, Lam);
        }
    }

    auto const& lam() const { return lam_; }
    auto const& mu() const { return mu_; }

    auto make_local_operator(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                             DGMethod method) const {
        auto elasticity =
            std::make_shared<Elasticity>(std::move(cl), lam_, mu_, std::nullopt, method);
        set(*elasticity);
        return elasticity;
    }

private:
    tagged_functional_t<1> lam_ =
        [](std::array<double, DomainDimension+1> const& v) -> std::array<double, 1> { return {1.0}; };
    tagged_functional_t<1> mu_ =
        [](std::array<double, DomainDimension+1> const& v) -> std::array<double, 1> { return {1.0}; };
};

} // namespace tndm

#endif // ELASTICITYSCENARIO_20200930_H
