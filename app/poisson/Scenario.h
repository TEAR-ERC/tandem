#ifndef SCENARIO_20200627_H
#define SCENARIO_20200627_H

#include "Poisson.h"
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
    std::optional<std::string> coefficient;
    std::optional<std::string> solution;
};

class Scenario {
public:
    using transform_t = Curvilinear<DomainDimension>::transform_t;
    using functional_t = Poisson::functional_t;

    virtual ~Scenario() {}
    virtual transform_t transform() const = 0;
    virtual functional_t force() const = 0;
    virtual functional_t boundary() const = 0;
    virtual functional_t slip() const = 0;
    virtual std::unique_ptr<SolutionInterface> solution() const = 0;
    virtual functional_t coefficient() const = 0;
};

class LuaScenario : public Scenario {
public:
    using solution_t = std::function<std::array<double, 1>(Vector<double> const&)>;

    LuaScenario(ProblemConfig const& problem) {
        lib_.loadFile(problem.lib);

        if (problem.warp) {
            warp_ = lib_.getFunction<DomainDimension, DomainDimension>(*problem.warp);
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
        functional(lib_, problem.force, force_);
        functional(lib_, problem.boundary, boundary_);
        functional(lib_, problem.slip, slip_);
        functional(lib_, problem.coefficient, coefficient_);
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

    Curvilinear<DomainDimension>::transform_t transform() const override { return warp_; }
    functional_t force() const override { return force_; }
    functional_t boundary() const override { return boundary_; }
    functional_t slip() const override { return slip_; }
    std::unique_ptr<SolutionInterface> solution() const override {
        return std::make_unique<LambdaSolution<decltype(solution_)>>(solution_);
    }
    functional_t coefficient() const override { return coefficient_; }

private:
    LuaLib lib_;
    transform_t warp_ = [](std::array<double, DomainDimension> const& v) { return v; };
    functional_t force_ = [](std::array<double, DomainDimension> const& v) { return 0.0; };
    functional_t boundary_ = [](std::array<double, DomainDimension> const& v) { return 0.0; };
    functional_t slip_ = [](std::array<double, DomainDimension> const& v) { return 0.0; };
    functional_t coefficient_ = [](std::array<double, DomainDimension> const& v) { return 1.0; };
    solution_t solution_ = [](Vector<double> const&) -> std::array<double, 1> { return {0.0}; };
};

} // namespace tndm

#endif // SCENARIO_20200627_H
