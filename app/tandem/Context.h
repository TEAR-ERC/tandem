#ifndef CONTEXT_20210910_H
#define CONTEXT_20210910_H

#include "config.h"
#include "form/AbstractAdapterOperator.h"
#include "form/AbstractFrictionOperator.h"
#include "form/AdapterOperator.h"
#include "form/FrictionOperator.h"
#include "form/SeasQDOperator.h"
#include "localoperator/DieterichRuinaAging.h"
#include "localoperator/DieterichRuinaSlip.h"
#include "localoperator/Elasticity.h"
#include "localoperator/Poisson.h"
#include "localoperator/RateAndState.h"
#include "tandem/ContextBase.h"
#include "tandem/FrictionConfig.h"
#include "tandem/SeasScenario.h"

#include "form/AbstractDGOperator.h"
#include "form/DGOperator.h"

#include <petscsys.h>

#include <array>
#include <optional>

namespace tndm::seas::detail {

template <typename Type> struct make_lop;
template <> struct make_lop<Poisson> {
    static auto dg(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                   SeasScenario<Poisson> const& scenario) {
        return std::make_shared<Poisson>(std::move(cl), scenario.mu(), DGMethod::IP);
    }
};
template <> struct make_lop<Elasticity> {
    static auto dg(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                   SeasScenario<Elasticity> const& scenario) {
        return std::make_shared<Elasticity>(std::move(cl), scenario.lam(), scenario.mu(),
                                            scenario.rho(), DGMethod::IP);
    }
};

} // namespace tndm::seas::detail

namespace tndm::seas {

template <typename Type> class Context : public ContextBase {
public:
    using adapter_t = AdapterOperator<Type>;
    using dg_t = DGOperator<Type>;
    #if defined(DR_AGING_LAW)
        using friction_lop_t = RateAndState<DieterichRuinaAging>;
    #elif defined(DR_SLIP_LAW)
        using friction_lop_t = RateAndState<DieterichRuinaSlip>;
    #endif
    using friction_t = FrictionOperator<friction_lop_t>;

    Context(LocalSimplexMesh<DomainDimension> const& mesh,
            std::unique_ptr<SeasScenario<Type>> seas_sc,
            std::unique_ptr<DieterichRuinaScenario> friction_sc,
            std::array<double, DomainDimension> up, std::array<double, DomainDimension> ref_normal)
        : ContextBase(mesh, seas_sc->transform()), scenario(std::move(seas_sc)),
          friction_scenario(std::move(friction_sc)),
          dg_lop(detail::make_lop<Type>::dg(cl, *scenario)), up(up), ref_normal(ref_normal) {}

    auto dg() -> std::unique_ptr<AbstractDGOperator<DomainDimension>> override {
        return std::make_unique<dg_t>(topo, dg_lop);
    }
    auto friction() -> std::unique_ptr<AbstractFrictionOperator> override {
        auto fric =
            std::make_unique<friction_t>(std::make_unique<friction_lop_t>(cl), topo, fault_map);
        fric->lop().set_constant_params(friction_scenario->constant_params());
        fric->lop().set_params(friction_scenario->param_fun());
        if (friction_scenario->source_fun()) {
            fric->lop().set_source_fun(*friction_scenario->source_fun());
        }
        if (friction_scenario->delta_tau_fun()) {
            fric->lop().set_delta_tau_fun(*friction_scenario->delta_tau_fun());
        }
        if (friction_scenario->delta_sn_fun()) {
            fric->lop().set_delta_sn_fun(*friction_scenario->delta_sn_fun());
        }
        return fric;
    }
    auto adapter() -> std::unique_ptr<AbstractAdapterOperator> override {
        return std::make_unique<adapter_t>(
            dg_lop,
            std::make_unique<Adapter<Type>>(cl, friction_lop_t::Space(),
                                            dg_lop->facetQuadratureRule(), up, ref_normal),
            topo, fault_map);
    }
    void setup_seasop(SeasQDOperator& seasop) override {
        if (scenario->boundary()) {
            seasop.set_boundary(std::make_unique<FacetFunctionalFactory<Type>>(
                dg_lop, *scenario->boundary(), ref_normal));
        }
    }
    void setup_seasop(SeasFDOperator& seasop) override {
        if (scenario->boundary()) {
            seasop.set_boundary(std::make_unique<FacetFunctionalFactory<Type>>(
                dg_lop, *scenario->boundary(), ref_normal));
        }
        if (scenario->initial_displacement()) {
            seasop.set_initial_displacement(std::make_unique<VolumeFunctionalFactory<Type>>(
                dg_lop, *scenario->initial_displacement()));
        }
        if (scenario->initial_velocity()) {
            seasop.set_initial_velocity(std::make_unique<VolumeFunctionalFactory<Type>>(
                dg_lop, *scenario->initial_velocity()));
        }
    }
    auto domain_solution(double time) -> std::unique_ptr<SolutionInterface> override {
        return scenario->solution(time);
    }
    auto fault_solution(double time) -> std::unique_ptr<SolutionInterface> override {
        return friction_scenario->solution(time);
    }

    std::unique_ptr<SeasScenario<Type>> scenario;
    std::unique_ptr<DieterichRuinaScenario> friction_scenario;
    std::shared_ptr<Type> dg_lop;

private:
    std::array<double, DomainDimension> up;
    std::array<double, DomainDimension> ref_normal;
};

} // namespace tndm::seas

#endif // CONTEXT_20210910_H
