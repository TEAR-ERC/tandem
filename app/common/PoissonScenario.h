#ifndef POISSONSCENARIO_20200930_H
#define POISSONSCENARIO_20200930_H

#include "common/Scenario.h"
#include "config.h"
#include "form/DGCurvilinearCommon.h"
#include "localoperator/Poisson.h"

#include <memory>
#include <optional>
#include <string>

namespace tndm {

class PoissonScenario : public Scenario<Poisson> {
public:
    constexpr static char Mu[] = "mu";

    PoissonScenario(LocalSimplexMesh<DomainDimension> const& mesh, std::string const& lib, std::string const& scenario,
                    std::array<double, DomainDimension> const& ref_normal)
        : Scenario(mesh, lib, scenario, ref_normal) {
        if (lib_.hasMember(scenario, Mu)) {
            coefficient_ = lib_.getMemberFunction<DomainDimension+1, 1>(scenario, Mu);
        }
    }

    auto const& coefficient() const { return coefficient_; }

    auto make_local_operator(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                             DGMethod method) const {
        auto poisson = std::make_shared<Poisson>(std::move(cl), coefficient_, method);
        set(*poisson);
        return poisson;
    }

private:
    tagged_functional_t<1> coefficient_ =
        [](std::array<double, DomainDimension+1> const& v) -> std::array<double, 1> { return {1.0}; };
};

} // namespace tndm

#endif // POISSONSCENARIO_20200930_H
