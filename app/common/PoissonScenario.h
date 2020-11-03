#ifndef POISSONSCENARIO_20200930_H
#define POISSONSCENARIO_20200930_H

#include "common/Scenario.h"
#include "config.h"
#include "localoperator/Poisson.h"

#include "util/Schema.h"

#include <memory>
#include <optional>
#include <string>

namespace tndm {

struct PoissonScenarioConfig : ScenarioConfig {
    std::optional<std::string> coefficient;

    template <typename PathConverter>
    static void setSchema(TableSchema<PoissonScenarioConfig>& schema,
                          PathConverter path_converter) {
        ScenarioConfig::setSchema(schema, std::move(path_converter));
        schema.add_value("coefficient", &PoissonScenarioConfig::coefficient);
    }
};

class PoissonScenario : public Scenario<Poisson> {
public:
    PoissonScenario(PoissonScenarioConfig const& problem) : Scenario(problem) {
        if (problem.coefficient) {
            coefficient_ = lib_.getFunction<DomainDimension, 1>(*problem.coefficient);
        }
    }

    auto const& coefficient() const { return coefficient_; }

    auto make_local_operator(std::shared_ptr<Curvilinear<DomainDimension>> cl) const {
        auto poisson = std::make_unique<Poisson>(std::move(cl), coefficient_);
        set(*poisson);
        return poisson;
    }

private:
    functional_t<1> coefficient_ =
        [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> { return {1.0}; };
};

} // namespace tndm

#endif // POISSONSCENARIO_20200930_H
