#ifndef SCENARIO_20200627_H
#define SCENARIO_20200627_H

#include "Elasticity.h"
#include "config.h"

#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "script/LuaLib.h"

namespace tndm {

struct ProblemConfig {
    std::string lib;
    std::optional<std::string> warp;
    std::optional<std::string> force;
    std::optional<std::string> boundary;
    std::optional<std::string> slip;
    std::optional<std::string> lam;
    std::optional<std::string> mu;
    std::optional<std::string> solution;
};

class LuaScenario {
public:
    using solution_t = std::function<std::array<double, DomainDimension>(Vector<double> const&)>;
    using transform_t = Curvilinear<DomainDimension>::transform_t;
    using functional_t = Elasticity::functional_t;
    using vector_functional_t = Elasticity::vector_functional_t;

    LuaScenario(ProblemConfig const& problem) {
        lib_.loadFile(problem.lib);

        if (problem.warp) {
            warp_ = lib_.getFunction<DomainDimension, DomainDimension>(*problem.warp);
        }
        if (problem.force) {
            force_ = lib_.getFunction<DomainDimension, DomainDimension>(*problem.force);
        }
        if (problem.boundary) {
            boundary_ = lib_.getFunction<DomainDimension, DomainDimension>(*problem.boundary);
        }
        if (problem.slip) {
            slip_ = lib_.getFunction<DomainDimension, DomainDimension>(*problem.slip);
        }
        auto functional = [](LuaLib& lib, std::optional<std::string> const& opt,
                             functional_t& target) {
            if (opt) {
                auto myF = lib.getFunction<DomainDimension, 1>(*opt);
                target = [myF](std::array<double, DomainDimension> const& v) -> double {
                    return myF(v)[0];
                };
            }
        };
        functional(lib_, problem.lam, lam_);
        functional(lib_, problem.mu, mu_);
        if (problem.solution) {
            auto myF = lib_.getFunction<DomainDimension, DomainDimension>(*problem.solution);
            solution_ = std::optional<solution_t>([myF](Vector<double> const& v) {
                std::array<double, DomainDimension> x;
                for (std::size_t i = 0; i < DomainDimension; ++i) {
                    x[i] = v(i);
                }
                return myF(x);
            });
        }
    }

    Curvilinear<DomainDimension>::transform_t transform() const { return warp_; }
    vector_functional_t force() const { return force_; }
    vector_functional_t boundary() const { return boundary_; }
    vector_functional_t slip() const { return slip_; }
    functional_t lam() const { return lam_; }
    functional_t mu() const { return mu_; }
    std::unique_ptr<SolutionInterface> solution() const {
        if (solution_) {
            return std::make_unique<LambdaSolution<decltype(*solution_)>>(*solution_);
        }
        return nullptr;
    }

private:
    LuaLib lib_;
    transform_t warp_ = [](std::array<double, DomainDimension> const& v) { return v; };
    vector_functional_t force_ = [](std::array<double, DomainDimension> const& v) {
        return std::array<double, DomainDimension>{};
    };
    vector_functional_t boundary_ = [](std::array<double, DomainDimension> const& v) {
        return std::array<double, DomainDimension>{};
    };
    vector_functional_t slip_ = [](std::array<double, DomainDimension> const& v) {
        return std::array<double, DomainDimension>{};
    };
    functional_t lam_ = [](std::array<double, DomainDimension> const& v) { return 1.0; };
    functional_t mu_ = [](std::array<double, DomainDimension> const& v) { return 1.0; };
    std::optional<solution_t> solution_ = std::nullopt;
};

} // namespace tndm

#endif // SCENARIO_20200627_H
