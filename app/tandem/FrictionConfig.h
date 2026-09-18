#ifndef FRICTIONCONFIG_20201027_H
#define FRICTIONCONFIG_20201027_H

#include "config.h"
#include "localoperator/DieterichRuinaBase.h"
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

class DieterichRuinaScenario {
public:
    template <std::size_t D>
    using functional_t =
        std::function<std::array<double, 1>(std::array<double, D> const&, long int&)>;
    template <std::size_t D>
    using vector_functional_t =
        std::function<std::array<double, DieterichRuinaBase::TangentialComponents>(
            std::array<double, D> const&, long int&)>;
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
    constexpr static char DeltaSn[] = "delta_sn";
    constexpr static char FaultSolution[] = "fault_solution";

    DieterichRuinaScenario(std::string const& lib, std::string const& scenario) {
        lib_.loadFile(lib);

        a_ = lib_.getMemberFunctionTagged<DomainDimension, 1>(scenario, A);
        eta_ = lib_.getMemberFunctionTagged<DomainDimension, 1>(scenario, Eta);
        L_ = lib_.getMemberFunctionTagged<DomainDimension, 1>(scenario, L);
        if (lib_.hasMember(scenario, SnPre)) {
            sn_pre_ = lib_.getMemberFunctionTagged<DomainDimension, 1>(scenario, SnPre);
        }
        if (lib_.hasMember(scenario, TauPre)) {
            tau_pre_ = lib_.getMemberFunctionTagged<DomainDimension,
                                                    DieterichRuinaBase::TangentialComponents>(
                scenario, TauPre);
        }
        Vinit_ =
            lib_.getMemberFunctionTagged<DomainDimension, DieterichRuinaBase::TangentialComponents>(
                scenario, Vinit);
        if (lib_.hasMember(scenario, Sinit)) {
            Sinit_ = lib_.getMemberFunctionTagged<DomainDimension,
                                                  DieterichRuinaBase::TangentialComponents>(
                scenario, Sinit);
        }
        if (lib_.hasMember(scenario, Source)) {
            source_ = std::make_optional(
                lib_.getMemberFunctionTagged<DomainDimension + 1, 1>(scenario, Source));
        }
        if (lib_.hasMember(scenario, DeltaTau)) {
            delta_tau_ = std::make_optional(
                lib_.getMemberFunctionTagged<DomainDimension + 1,
                                             DieterichRuinaBase::TangentialComponents>(scenario,
                                                                                       DeltaTau));
        }
        if (lib_.hasMember(scenario, DeltaSn)) {
            delta_sn_ = std::make_optional(
                lib_.getMemberFunctionTagged<DomainDimension + 1, 1>(scenario, DeltaSn));
        }

        cp_.V0 = lib_.getMemberConstant(scenario, V0);
        cp_.b = lib_.getMemberConstant(scenario, B);
        cp_.f0 = lib_.getMemberConstant(scenario, F0);

        if (lib_.hasMember(scenario, FaultSolution)) {
            solution_ = std::make_optional(SeasSolution<NumQuantities>(
                lib_.getMemberFunctionTagged<DomainDimension + 1, NumQuantities>(scenario,
                                                                                 FaultSolution)));
        }
    }

    auto const& constant_params() const { return cp_; }
    auto param_fun() const {
        return [this](std::array<double, DomainDimension> const& x, long int& region) {
            DieterichRuinaBase::Params p;
            p.a = this->a_(x, region)[0];
            p.eta = this->eta_(x, region)[0];
            p.L = this->L_(x, region)[0];
            p.sn_pre = this->sn_pre_(x, region)[0];
            p.tau_pre = this->tau_pre_(x, region);
            p.Vinit = this->Vinit_(x, region);
            p.Sinit = this->Sinit_(x, region);
            return p;
        };
    }

    auto const& source_fun() const { return source_; }
    auto const& delta_tau_fun() const { return delta_tau_; }
    auto const& delta_sn_fun() const { return delta_sn_; }
    std::unique_ptr<SolutionInterface> solution(double time) const {
        if (solution_) {
            auto sol = *solution_;
            sol.set_time(time);
            return std::make_unique<LambdaSolution<decltype(sol)>>(std::move(sol));
        }
        return nullptr;
    }

protected:
    DieterichRuinaBase::ConstantParams cp_;
    LuaLib lib_;
    functional_t<DomainDimension> a_, eta_, L_;
    functional_t<DomainDimension> sn_pre_ = [](std::array<double, DomainDimension> const& x,
                                               long int&) -> std::array<double, 1> {
        return {0.0};
    };
    vector_functional_t<DomainDimension> tau_pre_ =
        [](std::array<double, DomainDimension> const& x,
           long int&) -> std::array<double, DieterichRuinaBase::TangentialComponents> {
        return {};
    };
    vector_functional_t<DomainDimension> Vinit_;
    vector_functional_t<DomainDimension> Sinit_ =
        [](std::array<double, DomainDimension> const& x,
           long int&) -> std::array<double, DieterichRuinaBase::TangentialComponents> {
        return {};
    };
    std::optional<functional_t<DomainDimension + 1>> source_ = std::nullopt;
    std::optional<vector_functional_t<DomainDimension + 1>> delta_tau_ = std::nullopt;
    std::optional<functional_t<DomainDimension + 1>> delta_sn_ = std::nullopt;
    std::optional<SeasSolution<NumQuantities>> solution_ = std::nullopt;
};

} // namespace tndm

#endif // FRICTIONCONFIG_20201027_H
