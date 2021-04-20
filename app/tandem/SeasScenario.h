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

    constexpr static char Warp[] = "warp";
    constexpr static char Mu[] = "mu";
    constexpr static char Lam[] = "lam";
    constexpr static char Boundary[] = "boundary";
    constexpr static char Solution[] = "solution";

    SeasScenario(std::string const& lib, std::string const& scenario) {
        lib_.loadFile(lib);

        if (lib_.hasMember(scenario, Warp)) {
            warp_ = lib_.getMemberFunction<DomainDimension, DomainDimension>(scenario, Warp);
        }
        if (lib_.hasMember(scenario, Mu)) {
            mu_ = lib_.getMemberFunction<DomainDimension, 1>(scenario, Mu);
        }
        if (lib_.hasMember(scenario, Lam)) {
            lam_ = lib_.getMemberFunction<DomainDimension, 1>(scenario, Lam);
        }

        if (lib_.hasMember(scenario, Boundary)) {
            boundary_ = std::make_optional(
                lib_.getMemberFunction<DomainDimension + 1u, NumQuantities>(scenario, Boundary));
        }

        if (lib_.hasMember(scenario, Solution)) {
            solution_ = std::make_optional(SeasSolution<NumQuantities>(
                lib_.getMemberFunction<DomainDimension + 1, NumQuantities>(scenario, Solution)));
        }
    }

    auto const& transform() const { return warp_; }
    auto const& mu() const { return mu_; }
    auto const& lam() const { return lam_; }
    auto const& boundary() const { return boundary_; }
    std::unique_ptr<SolutionInterface> solution(double time) const {
        if (solution_) {
            auto sol = *solution_;
            sol.set_time(time);
            return std::make_unique<LambdaSolution<decltype(sol)>>(std::move(sol));
        }
        return nullptr;
    }

protected:
    LuaLib lib_;
    transform_t warp_ = [](std::array<double, DomainDimension> const& v) { return v; };
    functional_t mu_ = [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> {
        return {1.0};
    };
    functional_t lam_ = [](std::array<double, DomainDimension> const& v) -> std::array<double, 1> {
        return {0.0};
    };
    std::optional<time_functional_t> boundary_ = std::nullopt;
    std::optional<SeasSolution<NumQuantities>> solution_ = std::nullopt;
};

} // namespace tndm

#endif // SEASSCENARIO_20201006_H
