#ifndef ELASTICITYSCENARIO_20200930_H
#define ELASTICITYSCENARIO_20200930_H

#include "common/Scenario.h"
#include "config.h"
#include "form/DGCurvilinearCommon.h"
#include "localoperator/Elasticity.h"

#include "util/Schema.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace tndm {

struct ElasticityScenarioConfig : ScenarioConfig {
    std::optional<std::string> lam;
    std::optional<std::string> mu;

    template <typename PathConverter>
    static void setSchema(TableSchema<ElasticityScenarioConfig>& schema,
                          PathConverter path_converter) {
        ScenarioConfig::setSchema(schema, std::move(path_converter));
        schema.add_value("lam", &ElasticityScenarioConfig::lam);
        schema.add_value("mu", &ElasticityScenarioConfig::mu);
    }
};

class ElasticityScenario : public Scenario<Elasticity> {
public:
    ElasticityScenario(ElasticityScenarioConfig const& problem) : Scenario(problem) {
        if (problem.lam) {
            lam_ = lib_.getFunction<DomainDimension, 1>(*problem.lam);
        }
        if (problem.mu) {
            mu_ = lib_.getFunction<DomainDimension, 1>(*problem.mu);
        }
    }

    auto const& lam() const { return lam_; }
    auto const& mu() const { return mu_; }

    auto make_local_operator(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                             DGMethod method) const {
        auto elasticity = std::make_shared<Elasticity>(std::move(cl), lam_, mu_, method);
        set(*elasticity);
        return elasticity;
    }

private:
    functional_t<1> lam_ =
        [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> { return {1.0}; };
    functional_t<1> mu_ =
        [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> { return {1.0}; };
};

} // namespace tndm

#endif // ELASTICITYSCENARIO_20200930_H
