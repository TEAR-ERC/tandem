#ifndef SCENARIO_20201001_H
#define SCENARIO_20201001_H

#include "config.h"

#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "script/LuaLib.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"

#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace tndm {

struct ScenarioConfig {
    std::string lib;
    std::optional<std::string> warp;
    std::optional<std::string> force;
    std::optional<std::string> boundary;
    std::optional<std::string> slip;
    std::optional<std::string> solution;
    std::optional<std::array<double, DomainDimension>> ref_normal;

    template <typename Derived, typename PathConverter>
    static void setSchema(TableSchema<Derived>& schema, PathConverter path_converter) {
        auto cast = [](auto ptr) {
            using type = std::remove_reference_t<decltype(std::declval<ScenarioConfig>().*ptr)>;
            return static_cast<type Derived::*>(ptr);
        };
        schema.add_value("lib", cast(&Derived::lib))
            .converter(path_converter)
            .validator(PathExists());
        schema.add_value("warp", cast(&Derived::warp));
        schema.add_value("force", cast(&Derived::force));
        schema.add_value("boundary", cast(&Derived::boundary));
        schema.add_value("slip", cast(&Derived::slip));
        schema.add_value("solution", cast(&Derived::solution));
        schema.add_array("ref_normal", cast(&Derived::ref_normal)).of_values();
    }
};

template <class LocalOperator> class Scenario {
public:
    static constexpr std::size_t NumQuantities = LocalOperator::NumQuantities;
    using solution_t = std::function<std::array<double, NumQuantities>(Vector<double> const&)>;
    using transform_t = Curvilinear<DomainDimension>::transform_t;
    template <std::size_t Q> using functional_t = typename LocalOperator::template functional_t<Q>;

    Scenario(ScenarioConfig const& problem) : ref_normal_(problem.ref_normal) {
        lib_.loadFile(problem.lib);

        if (problem.warp) {
            warp_ = lib_.getFunction<DomainDimension, DomainDimension>(*problem.warp);
        }

        auto functional = [](LuaLib& lib, std::optional<std::string> const& opt,
                             std::optional<functional_t<NumQuantities>>& target) {
            if (opt) {
                target = std::make_optional(lib.getFunction<DomainDimension, NumQuantities>(*opt));
            }
        };
        functional(lib_, problem.force, force_);
        functional(lib_, problem.boundary, boundary_);
        functional(lib_, problem.slip, slip_);
        if (problem.solution) {
            auto myF = lib_.getFunction<DomainDimension, NumQuantities>(*problem.solution);
            solution_ = [myF](Vector<double> const& v) -> std::array<double, NumQuantities> {
                std::array<double, DomainDimension> x;
                for (std::size_t i = 0; i < DomainDimension; ++i) {
                    x[i] = v(i);
                }
                return myF(x);
            };
        }
    }

    auto const& transform() const { return warp_; }
    auto const& force() const { return force_; }
    auto const& boundary() const { return boundary_; }
    auto const& slip() const { return slip_; }
    std::unique_ptr<SolutionInterface> solution() const {
        if (solution_) {
            return std::make_unique<LambdaSolution<decltype(*solution_)>>(*solution_);
        }
        return nullptr;
    }

    void set(LocalOperator& lop) const {
        if (force_) {
            lop.set_force(*force_);
        }
        if (boundary_) {
            lop.set_dirichlet(*boundary_);
        }
        if (slip_ && ref_normal_) {
            lop.set_slip(*slip_, *ref_normal_);
        }
    }

protected:
    std::optional<std::array<double, DomainDimension>> ref_normal_;
    LuaLib lib_;
    transform_t warp_ = [](std::array<double, DomainDimension> const& v) { return v; };
    std::optional<functional_t<NumQuantities>> force_ = std::nullopt;
    std::optional<functional_t<NumQuantities>> boundary_ = std::nullopt;
    std::optional<functional_t<NumQuantities>> slip_ = std::nullopt;
    std::optional<solution_t> solution_ = std::nullopt;
};

} // namespace tndm

#endif // SCENARIO_20201001_H
