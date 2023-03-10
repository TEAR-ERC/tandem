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

template <class LocalOperator> class Scenario {
public:
    static constexpr std::size_t NumQuantities = LocalOperator::NumQuantities;
    using solution_t = std::function<std::array<double, NumQuantities>(Vector<double> const&)>;
    using solution_jacobian_t =
        std::function<std::array<double, NumQuantities * DomainDimension>(Vector<double> const&)>;
    using transform_t = Curvilinear<DomainDimension>::transform_t;
    // template <std::size_t Q> using functional_t = typename LocalOperator::template functional_t<Q>;
    template <std::size_t Q> using tagged_functional_t = typename LocalOperator::template tagged_functional_t<Q>;
	

    constexpr static char Warp[] = "warp";
    constexpr static char Force[] = "force";
    constexpr static char Boundary[] = "boundary";
    constexpr static char Slip[] = "slip";
    constexpr static char Solution[] = "solution";
    constexpr static char SolutionJacobian[] = "solution_jacobian";

    Scenario(LocalSimplexMesh<DomainDimension> const& mesh, std::string const& lib, std::string const& scenario,
             std::array<double, DomainDimension> const& ref_normal)
        : ref_normal_(ref_normal)/*,
		  ptags(dynamic_cast<ScalarMeshData<int> const*>(mesh.elements().pTagData())->getData()),
		  etags(dynamic_cast<ScalarMeshData<int> const*>(mesh.elements().eTagData())->getData())*/ {
        lib_.loadFile(lib);

        if (lib_.hasMember(scenario, Warp)) {
            warp_ = lib_.getMemberFunction<DomainDimension, DomainDimension>(scenario, Warp);
        }

        auto functional = [&](char const opt[],
                              std::optional<tagged_functional_t<NumQuantities>>& target) {
            if (lib_.hasMember(scenario, opt)) {
                target = std::make_optional(
                    lib_.getMemberFunction<DomainDimension+1u, NumQuantities>(scenario, opt));
            }
        };
        functional(Force, force_);
        functional(Boundary, boundary_);
        functional(Slip, slip_);
        if (lib_.hasMember(scenario, Solution)) {
            auto myF = lib_.getMemberFunction<DomainDimension, NumQuantities>(scenario, Solution);
            solution_ = [myF](Vector<double> const& v) -> std::array<double, NumQuantities> {
                std::array<double, DomainDimension> x;
                for (std::size_t i = 0; i < DomainDimension; ++i) {
                    x[i] = v(i);
                }
                return myF(x);
            };
        }
        if (lib_.hasMember(scenario, SolutionJacobian)) {
            auto myF = lib_.getMemberFunction<DomainDimension, NumQuantities * DomainDimension>(
                scenario, SolutionJacobian);
            solution_jacobian_ = [myF](Vector<double> const& v)
                -> std::array<double, NumQuantities * DomainDimension> {
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
    std::unique_ptr<SolutionInterface> solution_jacobian() const {
        if (solution_jacobian_) {
            return std::make_unique<LambdaSolution<decltype(*solution_jacobian_)>>(
                *solution_jacobian_);
        }
        return nullptr;
    }

    void set(LocalOperator& lop) const {
        if (force_) {
            lop.set_force(*force_);
        }
        if (boundary_) {
            lop.set_dirichlet(*boundary_, ref_normal_);
        }
        if (slip_) {
            lop.set_slip(*slip_, ref_normal_);
        }
    }

protected:
    std::array<double, DomainDimension> ref_normal_;
	// std::vector<int> const& ptags;
	// std::vector<int> const& etags;
    LuaLib lib_;
    transform_t warp_ = [](std::array<double, DomainDimension> const& v) { return v; };
    std::optional<tagged_functional_t<NumQuantities>> force_ = std::nullopt;
    std::optional<tagged_functional_t<NumQuantities>> boundary_ = std::nullopt;
    std::optional<tagged_functional_t<NumQuantities>> slip_ = std::nullopt;
    std::optional<solution_t> solution_ = std::nullopt;
    std::optional<solution_jacobian_t> solution_jacobian_ = std::nullopt;
};

} // namespace tndm

#endif // SCENARIO_20201001_H
