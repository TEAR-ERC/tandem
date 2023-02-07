#ifndef FRICTIONCONFIG_20201027_H
#define FRICTIONCONFIG_20201027_H

#include "config.h"
#include "localoperator/DieterichRuinaAgeing.h"
#include "localoperator/RateAndStateBase.h"
#include "tandem/SeasSolution.h"

#include "form/Error.h"
#include "script/LuaLib.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"

#include <array>
#include <functional>
#include <optional>
#include <string>

namespace tndm {

class DieterichRuinaAgeingScenario {
public:
    // template <std::size_t D>
    // using functional_t = std::function<std::array<double, 1>(std::array<double, D> const&)>;
    template <std::size_t D>
    using tagged_functional_t = std::function<std::array<double, 1>(std::array<double, D + 1> const&)>;
    // template <std::size_t D>
    // using vector_functional_t =
        // std::function<std::array<double, DieterichRuinaAgeing::TangentialComponents>(
            // std::array<double, D> const&)>;
    template <std::size_t D>
    using tagged_vector_functional_t =
        std::function<std::array<double, DieterichRuinaAgeing::TangentialComponents>(
            std::array<double, D + 1> const&)>;
    static constexpr std::size_t NumQuantities = RateAndStateBase::NumQuantities;

    constexpr static char A[] = "a";
    constexpr static char B[] = "b";
    constexpr static char V0[] = "V0";
    constexpr static char L[] = "L";
    constexpr static char F0[] = "f0";
    constexpr static char Eta[] = "eta";
    constexpr static char SnPre[] = "sn_pre";
    constexpr static char TauPre[] = "tau_pre";
    constexpr static char Vinit[] = "Vinit";
    constexpr static char Sinit[] = "Sinit";
    constexpr static char Source[] = "source";
    constexpr static char DeltaTau[] = "delta_tau";
    constexpr static char FaultSolution[] = "fault_solution";

    DieterichRuinaAgeingScenario(std::string const& lib, std::string const& scenario) {
        lib_.loadFile(lib);

        a_   = lib_.getMemberFunction<DomainDimension+1u, 1>(scenario, A);
		b_   = lib_.getMemberFunction<DomainDimension+1u, 1>(scenario, B);
        eta_ = lib_.getMemberFunction<DomainDimension+1u, 1>(scenario, Eta);
        L_   = lib_.getMemberFunction<DomainDimension+1u, 1>(scenario, L);
        if (lib_.hasMember(scenario, SnPre)) {
            sn_pre_ = lib_.getMemberFunction<DomainDimension+1u, 1>(scenario, SnPre);
        }
        if (lib_.hasMember(scenario, TauPre)) {
            tau_pre_ =
                lib_.getMemberFunction<DomainDimension+1u, DieterichRuinaAgeing::TangentialComponents>(
                    scenario, TauPre);
        }
        Vinit_ =
            lib_.getMemberFunction<DomainDimension+1u, DieterichRuinaAgeing::TangentialComponents>(
                scenario, Vinit);
        if (lib_.hasMember(scenario, Sinit)) {
            Sinit_ =
                lib_.getMemberFunction<DomainDimension+1u, DieterichRuinaAgeing::TangentialComponents>(
                    scenario, Sinit);
        }
        if (lib_.hasMember(scenario, Source)) {
            source_ = std::make_optional(
                lib_.getMemberFunction<DomainDimension+2u, 1>(scenario, Source));
        }
        if (lib_.hasMember(scenario, DeltaTau)) {
            delta_tau_ = std::make_optional(
                lib_.getMemberFunction<DomainDimension+2u,
                                       DieterichRuinaAgeing::TangentialComponents>(scenario,
                                                                                   DeltaTau));
        }

        cp_.V0 = lib_.getMemberConstant(scenario, V0);
        // cp_.b = lib_.getMemberConstant(scenario, B);
        cp_.f0 = lib_.getMemberConstant(scenario, F0);

        if (lib_.hasMember(scenario, FaultSolution)) {
            solution_ = std::make_optional(SeasSolution<NumQuantities>(
                lib_.getMemberFunction<DomainDimension + 1u, NumQuantities>(scenario,
                                                                           FaultSolution)));
        }
    }

    auto const& constant_params() const { return cp_; }
    auto param_fun() const {
        return [this](std::array<double, DomainDimension+1> const& x) {
            DieterichRuinaAgeing::Params p;
            p.a = this->a_(x)[0];
            p.b = this->b_(x)[0];
            p.eta = this->eta_(x)[0];
            p.L = this->L_(x)[0];
            p.sn_pre = this->sn_pre_(x)[0];
            p.tau_pre = this->tau_pre_(x);
            p.Vinit = this->Vinit_(x);
            p.Sinit = this->Sinit_(x);
            return p;
        };
    }
    auto const& source_fun() const { return source_; }
    auto const& delta_tau_fun() const { return delta_tau_; }
    std::unique_ptr<SolutionInterface> solution(double time) const {
        if (solution_) {
            auto sol = *solution_;
            sol.set_time(time);
            return std::make_unique<LambdaSolution<decltype(sol)>>(std::move(sol));
        }
        return nullptr;
    }

protected:
    DieterichRuinaAgeing::ConstantParams cp_;
    LuaLib lib_;
    tagged_functional_t<DomainDimension> a_, b_, eta_, L_;
    tagged_functional_t<DomainDimension> sn_pre_ =
        [](std::array<double, DomainDimension+1u> const& x) -> std::array<double, 1> { return {0.0}; };
    tagged_vector_functional_t<DomainDimension> tau_pre_ = [](std::array<double, DomainDimension+1u> const& x)
        -> std::array<double, DieterichRuinaAgeing::TangentialComponents> { return {}; };
    tagged_vector_functional_t<DomainDimension> Vinit_;
    tagged_vector_functional_t<DomainDimension> Sinit_ = [](std::array<double, DomainDimension+1u> const& x)
        -> std::array<double, DieterichRuinaAgeing::TangentialComponents> { return {}; };
    std::optional<tagged_functional_t<DomainDimension + 1>> source_ = std::nullopt;
    std::optional<tagged_vector_functional_t<DomainDimension + 1>> delta_tau_ = std::nullopt;
    std::optional<SeasSolution<NumQuantities>> solution_ = std::nullopt;
};

} // namespace tndm

#endif // FRICTIONCONFIG_20201027_H
