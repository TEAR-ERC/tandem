#ifndef SCENARIO_20200627_H
#define SCENARIO_20200627_H

#include "config.h"
#include "localoperator/Poisson.h"

#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "script/LuaLib.h"

#include <optional>

namespace tndm {

struct PoissonScenarioConfig {
    std::string lib;
    std::optional<std::string> warp;
    std::optional<std::string> force;
    std::optional<std::string> boundary;
    std::optional<std::string> slip;
    std::optional<std::string> coefficient;
    std::optional<std::string> solution;
    std::optional<std::array<double, DomainDimension>> ref_normal;
};

class PoissonScenario {
public:
    using solution_t = std::function<std::array<double, 1>(Vector<double> const&)>;
    using transform_t = Curvilinear<DomainDimension>::transform_t;
    template <std::size_t Q> using functional_t = tmp::Poisson::functional_t<Q>;

    PoissonScenario(PoissonScenarioConfig const& problem) {
        lib_.loadFile(problem.lib);

        if (problem.warp) {
            warp_ = lib_.getFunction<DomainDimension, DomainDimension>(*problem.warp);
        }
        if (problem.coefficient) {
            coefficient_ = lib_.getFunction<DomainDimension, 1>(*problem.coefficient);
        }
        auto functional = [](LuaLib& lib, std::optional<std::string> const& opt,
                             std::optional<functional_t<1>>& target) {
            if (opt) {
                target = std::make_optional(lib.getFunction<DomainDimension, 1>(*opt));
            }
        };
        functional(lib_, problem.force, force_);
        functional(lib_, problem.boundary, boundary_);
        functional(lib_, problem.slip, slip_);
        if (problem.solution) {
            auto myF = lib_.getFunction<DomainDimension, 1>(*problem.solution);
            solution_ = [myF](Vector<double> const& v) -> std::array<double, 1> {
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
    auto const& coefficient() const { return coefficient_; }

private:
    LuaLib lib_;
    transform_t warp_ = [](std::array<double, DomainDimension> const& v) { return v; };
    functional_t<1> coefficient_ =
        [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> { return {1.0}; };
    std::optional<functional_t<1>> force_ = std::nullopt;
    std::optional<functional_t<1>> boundary_ = std::nullopt;
    std::optional<functional_t<1>> slip_ = std::nullopt;
    std::optional<solution_t> solution_ = std::nullopt;
};

} // namespace tndm

#endif // SCENARIO_20200627_H
