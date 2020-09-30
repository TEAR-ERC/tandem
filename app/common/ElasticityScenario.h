#ifndef ELASTICITYSCENARIO_20200930_H
#define ELASTICITYSCENARIO_20200930_H

#include "config.h"
#include "localoperator/Elasticity.h"

#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "script/LuaLib.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"

#include <Eigen/src/Core/util/Macros.h>
#include <optional>
#include <string>

namespace tndm {

struct ElasticityScenarioConfig {
    std::string lib;
    std::optional<std::string> warp;
    std::optional<std::string> force;
    std::optional<std::string> boundary;
    std::optional<std::string> slip;
    std::optional<std::string> lam;
    std::optional<std::string> mu;
    std::optional<std::string> solution;
    std::optional<std::array<double, DomainDimension>> ref_normal;

    template <typename PathConverter>
    static void setSchema(TableSchema<ElasticityScenarioConfig>& schema,
                          PathConverter path_converter) {
        schema.add_value("lib", &ElasticityScenarioConfig::lib)
            .converter(path_converter)
            .validator(PathExists());
        schema.add_value("warp", &ElasticityScenarioConfig::warp);
        schema.add_value("force", &ElasticityScenarioConfig::force);
        schema.add_value("boundary", &ElasticityScenarioConfig::boundary);
        schema.add_value("slip", &ElasticityScenarioConfig::slip);
        schema.add_value("lam", &ElasticityScenarioConfig::lam);
        schema.add_value("mu", &ElasticityScenarioConfig::mu);
        schema.add_value("solution", &ElasticityScenarioConfig::solution);
        schema.add_array("ref_normal", &ElasticityScenarioConfig::ref_normal).of_values();
    }
};

class ElasticityScenario {
public:
    using solution_t = std::function<std::array<double, 1>(Vector<double> const&)>;
    using transform_t = Curvilinear<DomainDimension>::transform_t;
    template <std::size_t Q> using functional_t = tmp::Elasticity::functional_t<Q>;
    static constexpr std::size_t NumQuantities = tmp::Elasticity::NumQuantities;

    ElasticityScenario(ElasticityScenarioConfig const& problem) : ref_normal_(problem.ref_normal) {
        lib_.loadFile(problem.lib);

        if (problem.warp) {
            warp_ = lib_.getFunction<DomainDimension, DomainDimension>(*problem.warp);
        }
        if (problem.lam) {
            lam_ = lib_.getFunction<DomainDimension, 1>(*problem.lam);
        }
        if (problem.mu) {
            mu_ = lib_.getFunction<DomainDimension, 1>(*problem.mu);
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
    auto const& lam() const { return lam_; }
    auto const& mu() const { return mu_; }

    auto make_local_operator(Curvilinear<DomainDimension> const& cl) const {
        auto poisson = std::make_unique<tmp::Elasticity>(cl, lam_, mu_);
        if (force_) {
            poisson->set_force(*force_);
        }
        if (boundary_) {
            poisson->set_dirichlet(*boundary_);
        }
        if (slip_ && ref_normal_) {
            poisson->set_slip(*slip_, *ref_normal_);
        }
        return poisson;
    }

private:
    std::optional<std::array<double, DomainDimension>> ref_normal_;
    LuaLib lib_;
    transform_t warp_ = [](std::array<double, DomainDimension> const& v) { return v; };
    functional_t<1> lam_ =
        [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> { return {1.0}; };
    functional_t<1> mu_ =
        [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> { return {1.0}; };
    std::optional<functional_t<NumQuantities>> force_ = std::nullopt;
    std::optional<functional_t<NumQuantities>> boundary_ = std::nullopt;
    std::optional<functional_t<NumQuantities>> slip_ = std::nullopt;
    std::optional<solution_t> solution_ = std::nullopt;
};

} // namespace tndm

#endif // ELASTICITYSCENARIO_20200930_H
