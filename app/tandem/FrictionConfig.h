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

struct DieterichRuinaAgeingConfig {
    using Constant = DieterichRuinaAgeing::ConstantParams;
    std::string lib;
    std::string a;
    std::string eta;
    std::optional<std::string> tau_pre;
    std::string Vinit;
    std::optional<std::string> Sinit;
    std::optional<std::string> source;
    Constant constant;

    template <typename PathConverter>
    static void setSchema(TableSchema<DieterichRuinaAgeingConfig>& schema,
                          PathConverter path_converter) {
        schema.add_value("lib", &DieterichRuinaAgeingConfig::lib)
            .converter(path_converter)
            .validator(PathExists());
        schema.add_value("a", &DieterichRuinaAgeingConfig::a);
        schema.add_value("eta", &DieterichRuinaAgeingConfig::eta);
        schema.add_value("tau_pre", &DieterichRuinaAgeingConfig::tau_pre);
        schema.add_value("Vinit", &DieterichRuinaAgeingConfig::Vinit);
        schema.add_value("Sinit", &DieterichRuinaAgeingConfig::Sinit);
        schema.add_value("source", &DieterichRuinaAgeingConfig::source);
        auto& cs = schema.add_table("constant", &DieterichRuinaAgeingConfig::constant);
        cs.add_value("sn", &Constant::sn);
        cs.add_value("V0", &Constant::V0);
        cs.add_value("b", &Constant::b);
        cs.add_value("L", &Constant::L);
        cs.add_value("f0", &Constant::f0);
    }
};

class DieterichRuinaAgeingScenario {
public:
    template <std::size_t D>
    using functional_t = std::function<std::array<double, 1>(std::array<double, D> const&)>;

    DieterichRuinaAgeingScenario(DieterichRuinaAgeingConfig const& cfg) : cp_(cfg.constant) {
        lib_.loadFile(cfg.lib);

        a_ = lib_.getFunction<DomainDimension, 1>(cfg.a);
        eta_ = lib_.getFunction<DomainDimension, 1>(cfg.eta);
        if (cfg.tau_pre) {
            tau_pre_ = lib_.getFunction<DomainDimension, 1>(*cfg.tau_pre);
        }
        Vinit_ = lib_.getFunction<DomainDimension, 1>(cfg.Vinit);
        if (cfg.Sinit) {
            Sinit_ = lib_.getFunction<DomainDimension, 1>(*cfg.Sinit);
        }
        if (cfg.source) {
            source_ = std::make_optional(lib_.getFunction<DomainDimension + 1, 1>(*cfg.source));
        }
    }

    auto const& constant_params() const { return cp_; }
    auto param_fun() const {
        return [this](std::array<double, DomainDimension> const& x) {
            DieterichRuinaAgeing::Params p;
            p.a = this->a_(x)[0];
            p.eta = this->eta_(x)[0];
            p.tau_pre = this->tau_pre_(x)[0];
            p.Vinit = this->Vinit_(x)[0];
            p.Sinit = this->Sinit_(x)[0];
            return p;
        };
    }
    auto const& source_fun() const { return source_; }

protected:
    DieterichRuinaAgeing::ConstantParams cp_;
    LuaLib lib_;
    functional_t<DomainDimension> a_, eta_;
    functional_t<DomainDimension> tau_pre_ =
        [](std::array<double, DomainDimension> const& x) -> std::array<double, 1> { return {0.0}; };
    functional_t<DomainDimension> Vinit_;
    functional_t<DomainDimension> Sinit_ =
        [](std::array<double, DomainDimension> const& x) -> std::array<double, 1> { return {0.0}; };
    std::optional<functional_t<DomainDimension + 1>> source_;
};

} // namespace tndm

#endif // FRICTIONCONFIG_20201027_H
