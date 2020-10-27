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
    std::string lib;
    std::string a;
    std::string tau0;
    double eta;
    double Vinit;
    double sn;
    double V0;
    double b;
    double L;
    double f0;

    template <typename PathConverter>
    static void setSchema(TableSchema<DieterichRuinaAgeingConfig>& schema,
                          PathConverter path_converter) {
        schema.add_value("lib", &DieterichRuinaAgeingConfig::lib)
            .converter(path_converter)
            .validator(PathExists());
        schema.add_value("a", &DieterichRuinaAgeingConfig::a);
        schema.add_value("tau0", &DieterichRuinaAgeingConfig::tau0);
        schema.add_value("eta", &DieterichRuinaAgeingConfig::eta);
        schema.add_value("Vinit", &DieterichRuinaAgeingConfig::Vinit);
        schema.add_value("sn", &DieterichRuinaAgeingConfig::sn);
        schema.add_value("V0", &DieterichRuinaAgeingConfig::V0);
        schema.add_value("b", &DieterichRuinaAgeingConfig::b);
        schema.add_value("L", &DieterichRuinaAgeingConfig::L);
        schema.add_value("f0", &DieterichRuinaAgeingConfig::f0);
    }
};

class DieterichRuinaAgeingScenario {
public:
    using functional_t =
        std::function<std::array<double, 1>(std::array<double, DomainDimension> const&)>;

    DieterichRuinaAgeingScenario(DieterichRuinaAgeingConfig const& cfg) {
        lib_.loadFile(cfg.lib);

        a_ = lib_.getFunction<DomainDimension, 1>(cfg.a);
        tau0_ = lib_.getFunction<DomainDimension, 1>(cfg.tau0);

        cp_.eta = cfg.eta;
        cp_.Vinit = cfg.Vinit;
        cp_.sn = cfg.sn;
        cp_.V0 = cfg.V0;
        cp_.b = cfg.b;
        cp_.L = cfg.L;
        cp_.f0 = cfg.f0;
    }

    auto const& constant_params() const { return cp_; }
    auto param_fun() const {
        return [this](std::array<double, DomainDimension> const& x) {
            DieterichRuinaAgeing::Params p;
            p.a = this->a_(x)[0];
            p.tau0 = this->tau0_(x)[0];
            return p;
        };
    }

protected:
    LuaLib lib_;
    functional_t a_;
    functional_t tau0_;
    DieterichRuinaAgeing::ConstantParams cp_;
};

} // namespace tndm

#endif // FRICTIONCONFIG_20201027_H
