#ifndef SEASSCENARIO_20201006_H
#define SEASSCENARIO_20201006_H

#include "config.h"

#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "script/LuaLib.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace tndm {

struct SeasScenarioConfig {
    std::string lib;
    std::optional<std::string> warp;
    std::optional<std::string> mu;
    std::optional<std::string> boundary;
    std::array<double, DomainDimension> ref_normal;

    template <typename PathConverter>
    static void setSchema(TableSchema<SeasScenarioConfig>& schema, PathConverter path_converter) {
        schema.add_value("lib", &SeasScenarioConfig::lib)
            .converter(path_converter)
            .validator(PathExists());
        schema.add_value("warp", &SeasScenarioConfig::warp);
        schema.add_value("mu", &SeasScenarioConfig::mu);
        schema.add_value("boundary", &SeasScenarioConfig::boundary);
        schema.add_array("ref_normal", &SeasScenarioConfig::ref_normal).of_values();
    }
};

template <class LocalOperator> class SeasScenario {
public:
    static constexpr std::size_t NumQuantities = LocalOperator::NumQuantities;
    using transform_t = Curvilinear<DomainDimension>::transform_t;
    using functional_t =
        std::function<std::array<double, 1>(std::array<double, DomainDimension> const&)>;
    template <std::size_t Q>
    using time_functional_t =
        std::function<std::array<double, Q>(std::array<double, DomainDimension + 1u> const&)>;

    SeasScenario(SeasScenarioConfig const& problem) : ref_normal_(problem.ref_normal) {
        lib_.loadFile(problem.lib);

        if (problem.warp) {
            warp_ = lib_.getFunction<DomainDimension, DomainDimension>(*problem.warp);
        }
        if (problem.mu) {
            mu_ = lib_.getFunction<DomainDimension, 1>(*problem.mu);
        }

        auto time_functional = [](LuaLib& lib, std::optional<std::string> const& opt,
                                  std::optional<time_functional_t<NumQuantities>>& target) {
            if (opt) {
                target =
                    std::make_optional(lib.getFunction<DomainDimension + 1u, NumQuantities>(*opt));
            }
        };
        time_functional(lib_, problem.boundary, boundary_);
    }

    auto const& transform() const { return warp_; }
    auto const& mu() const { return mu_; }
    auto const& boundary() const { return boundary_; }
    auto const& ref_normal() const { return ref_normal_; }

protected:
    std::array<double, DomainDimension> ref_normal_;
    LuaLib lib_;
    transform_t warp_ = [](std::array<double, DomainDimension> const& v) { return v; };
    functional_t mu_ = [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> {
        return {1.0};
    };
    std::optional<time_functional_t<NumQuantities>> boundary_ = std::nullopt;
};

} // namespace tndm

#endif // SEASSCENARIO_20201006_H
