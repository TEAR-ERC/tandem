#ifndef FRICTIONCONFIG_20201027_H
#define FRICTIONCONFIG_20201027_H

#include "config.h"
#include "tandem/DieterichRuinaAgeing.h"

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
    template <std::size_t D>
    using functional_t = std::function<std::array<double, 1>(std::array<double, D> const&)>;
    template <std::size_t D>
    using vector_functional_t =
        std::function<std::array<double, DieterichRuinaAgeing::TangentialComponents>(
            std::array<double, D> const&)>;

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

    DieterichRuinaAgeingScenario(std::string const& lib, std::string const& scenario) {
        lib_.loadFile(lib);

        a_ = lib_.getMemberFunction<DomainDimension, 1>(scenario, A);
        eta_ = lib_.getMemberFunction<DomainDimension, 1>(scenario, Eta);
        L_ = lib_.getMemberFunction<DomainDimension, 1>(scenario, L);
        if (lib_.hasMember(scenario, SnPre)) {
            sn_pre_ = lib_.getMemberFunction<DomainDimension, 1>(scenario, SnPre);
        }
        if (lib_.hasMember(scenario, TauPre)) {
            tau_pre_ =
                lib_.getMemberFunction<DomainDimension, DieterichRuinaAgeing::TangentialComponents>(
                    scenario, TauPre);
        }
        Vinit_ =
            lib_.getMemberFunction<DomainDimension, DieterichRuinaAgeing::TangentialComponents>(
                scenario, Vinit);
        if (lib_.hasMember(scenario, Sinit)) {
            Sinit_ =
                lib_.getMemberFunction<DomainDimension, DieterichRuinaAgeing::TangentialComponents>(
                    scenario, Sinit);
        }
        if (lib_.hasMember(scenario, Source)) {
            source_ = std::make_optional(
                lib_.getMemberFunction<DomainDimension + 1, 1>(scenario, Source));
        }

        cp_.V0 = lib_.getMemberConstant(scenario, V0);
        cp_.b = lib_.getMemberConstant(scenario, B);
        cp_.f0 = lib_.getMemberConstant(scenario, F0);
    }

    auto const& constant_params() const { return cp_; }
    auto param_fun() const {
        return [this](std::array<double, DomainDimension> const& x) {
            DieterichRuinaAgeing::Params p;
            p.a = this->a_(x)[0];
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

protected:
    DieterichRuinaAgeing::ConstantParams cp_;
    LuaLib lib_;
    functional_t<DomainDimension> a_, eta_, L_;
    functional_t<DomainDimension> sn_pre_ =
        [](std::array<double, DomainDimension> const& x) -> std::array<double, 1> { return {0.0}; };
    vector_functional_t<DomainDimension> tau_pre_ = [](std::array<double, DomainDimension> const& x)
        -> std::array<double, DieterichRuinaAgeing::TangentialComponents> { return {}; };
    vector_functional_t<DomainDimension> Vinit_;
    vector_functional_t<DomainDimension> Sinit_ = [](std::array<double, DomainDimension> const& x)
        -> std::array<double, DieterichRuinaAgeing::TangentialComponents> { return {}; };
    std::optional<functional_t<DomainDimension + 1>> source_;
};

} // namespace tndm

#endif // FRICTIONCONFIG_20201027_H
