#ifndef SEASSCENARIO_20201006_H
#define SEASSCENARIO_20201006_H

#include "config.h"

#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "script/LuaLib.h"
#include "tandem/Elasticity.h"
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
    enum class Type { Unknown = 0, Poisson = 1, Elasticity = 2 };
    Type type;
    std::string lib;
    std::optional<std::string> warp;
    std::optional<std::string> mu;
    std::optional<std::string> lam;
    std::optional<std::string> boundary;
    std::optional<std::string> solution;
    std::array<double, DomainDimension> ref_normal;

    template <typename PathConverter>
    static void setSchema(TableSchema<SeasScenarioConfig>& schema, PathConverter path_converter) {
        schema.add_value("type", &SeasScenarioConfig::type)
            .converter([](std::string_view value) {
                if (iEquals(value, "poisson")) {
                    return Type::Poisson;
                } else if (iEquals(value, "elastic") || iEquals(value, "elasticity")) {
                    return Type::Elasticity;
                } else {
                    return Type::Unknown;
                }
            })
            .validator([](Type const& type) { return type != Type::Unknown; });
        schema.add_value("lib", &SeasScenarioConfig::lib)
            .converter(path_converter)
            .validator(PathExists());
        schema.add_value("warp", &SeasScenarioConfig::warp);
        schema.add_value("mu", &SeasScenarioConfig::mu);
        schema.add_value("boundary", &SeasScenarioConfig::boundary);
        schema.add_value("solution", &SeasScenarioConfig::solution);
        schema.add_array("ref_normal", &SeasScenarioConfig::ref_normal).of_values();
    }
};

template <std::size_t NumQuantities> class SeasSolution {
public:
    using time_functional_t = LuaLib::functional_t<DomainDimension + 1, NumQuantities>;

    SeasSolution(time_functional_t solution) : solution_(std::move(solution)) {}

    std::array<double, NumQuantities> operator()(Vector<double> const& v) const {
        std::array<double, DomainDimension + 1> xt;
        for (std::size_t i = 0; i < DomainDimension; ++i) {
            xt[i] = v(i);
        }
        xt.back() = time_;
        return solution_(xt);
    }

    void set_time(double time) { time_ = time; }

private:
    time_functional_t solution_;
    double time_ = 0.0;
};

template <class LocalOperator> class SeasScenario {
public:
    static constexpr std::size_t NumQuantities = LocalOperator::NumQuantities;
    using transform_t = Curvilinear<DomainDimension>::transform_t;
    using functional_t = LuaLib::functional_t<DomainDimension, 1>;
    using time_functional_t = LuaLib::functional_t<DomainDimension + 1, NumQuantities>;

    SeasScenario(SeasScenarioConfig const& problem) : ref_normal_(problem.ref_normal) {
        lib_.loadFile(problem.lib);

        if (problem.warp) {
            warp_ = lib_.getFunction<DomainDimension, DomainDimension>(*problem.warp);
        }
        if (problem.mu) {
            mu_ = lib_.getFunction<DomainDimension, 1>(*problem.mu);
        }

        auto time_functional = [](LuaLib& lib, std::optional<std::string> const& opt,
                                  std::optional<time_functional_t>& target) {
            if (opt) {
                target =
                    std::make_optional(lib.getFunction<DomainDimension + 1u, NumQuantities>(*opt));
            }
        };
        time_functional(lib_, problem.boundary, boundary_);

        if (problem.solution) {
            solution_ = std::make_optional(SeasSolution<NumQuantities>(
                lib_.getFunction<DomainDimension + 1, NumQuantities>(*problem.solution)));
        }
    }

    auto const& transform() const { return warp_; }
    auto const& mu() const { return mu_; }
    auto const& boundary() const { return boundary_; }
    auto const& ref_normal() const { return ref_normal_; }
    std::unique_ptr<SolutionInterface> solution(double time) const {
        if (solution_) {
            auto sol = *solution_;
            sol.set_time(time);
            return std::make_unique<LambdaSolution<decltype(sol)>>(std::move(sol));
        }
        return nullptr;
    }

protected:
    std::array<double, DomainDimension> ref_normal_;
    LuaLib lib_;
    transform_t warp_ = [](std::array<double, DomainDimension> const& v) { return v; };
    functional_t mu_ = [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> {
        return {1.0};
    };
    std::optional<time_functional_t> boundary_ = std::nullopt;
    std::optional<SeasSolution<NumQuantities>> solution_ = std::nullopt;
};

} // namespace tndm

#endif // SEASSCENARIO_20201006_H
