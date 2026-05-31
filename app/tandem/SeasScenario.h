#ifndef SEASSCENARIO_20201006_H
#define SEASSCENARIO_20201006_H

#include "config.h"
#include "tandem/SeasSolution.h"

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

template <class LocalOperator> class SeasScenario {
public:
    static constexpr std::size_t NumQuantities = LocalOperator::NumQuantities;
    using transform_t = Curvilinear<DomainDimension>::transform_t;
    using functional_t = LuaLib::functional_t<DomainDimension, 1>;
    using functional_t_region = LuaLib::functional_t_region<DomainDimension, 1>;
    using time_functional_t = LuaLib::functional_t<DomainDimension + 1, NumQuantities>;
    using vector_functional_t = LuaLib::functional_t<DomainDimension, NumQuantities>;

    constexpr static char Warp[] = "warp";
    constexpr static char Mu[] = "mu";
    constexpr static char Lam[] = "lam";
    constexpr static char Rho[] = "rho";
    constexpr static char Boundary[] = "boundary";
    constexpr static char Solution[] = "solution";
    constexpr static char InitialDisplacement[] = "initial_displacement";
    constexpr static char InitialVelocity[] = "initial_velocity";

    SeasScenario(std::string const& lib, std::string const& scenario) {
        lib_.loadFile(lib);

        if (lib_.hasMember(scenario, Warp)) {
            warp_ = lib_.getMemberFunction<DomainDimension, DomainDimension>(scenario, Warp);
        }
        if (lib_.hasMember(scenario, Mu)) {
            mu_ = lib_.getMemberFunctionTagged<DomainDimension, 1>(scenario, Mu);
        }
        if (lib_.hasMember(scenario, Lam)) {
            lam_ = lib_.getMemberFunctionTagged<DomainDimension, 1>(scenario, Lam);
        }
        if (lib_.hasMember(scenario, Rho)) {
            rho_ = std::make_optional(lib_.getMemberFunctionTagged<DomainDimension, 1>(scenario, Rho));
        }

        if (lib_.hasMember(scenario, Boundary)) {
            boundary_ = std::make_optional(
                lib_.getMemberFunction<DomainDimension + 1u, NumQuantities>(scenario, Boundary));
        }

        if (lib_.hasMember(scenario, Solution)) {
            solution_ = std::make_optional(SeasSolution<NumQuantities>(
                lib_.getMemberFunction<DomainDimension + 1, NumQuantities>(scenario, Solution)));
        }

        if (lib_.hasMember(scenario, InitialDisplacement)) {
            u_ini_ = std::make_optional(lib_.getMemberFunction<DomainDimension, NumQuantities>(
                scenario, InitialDisplacement));
        }

        if (lib_.hasMember(scenario, InitialVelocity)) {
            v_ini_ = std::make_optional(
                lib_.getMemberFunction<DomainDimension, NumQuantities>(scenario, InitialVelocity));
        }
    }

    auto const& transform() const { return warp_; }
    auto const& mu() const { return mu_; }
    auto const& lam() const { return lam_; }
    auto const& rho() const { return rho_; }
    auto const& boundary() const { return boundary_; }
    std::unique_ptr<SolutionInterface> solution(double time) const {
        if (solution_) {
            auto sol = *solution_;
            sol.set_time(time);
            return std::make_unique<LambdaSolution<decltype(sol)>>(std::move(sol));
        }
        return nullptr;
    }
    auto const& initial_displacement() const { return u_ini_; }
    auto const& initial_velocity() const { return v_ini_; }

protected:
    LuaLib lib_;
    transform_t warp_ = [](std::array<double, DomainDimension> const& v) { return v; };
    functional_t_region mu_ = [](std::array<double, DomainDimension> const& v, long int&) -> std::array<double, 1> {
        return {1.0};
    };
    functional_t_region lam_ = [](std::array<double, DomainDimension> const& v, long int&) -> std::array<double, 1> {
        return {0.0};
    };
    std::optional<functional_t_region> rho_ = std::nullopt;
    std::optional<time_functional_t> boundary_ = std::nullopt;
    std::optional<SeasSolution<NumQuantities>> solution_ = std::nullopt;
    std::optional<vector_functional_t> u_ini_ = std::nullopt;
    std::optional<vector_functional_t> v_ini_ = std::nullopt;
};

} // namespace tndm

#endif // SEASSCENARIO_20201006_H
