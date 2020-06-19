#ifndef TABULATEDRULES_20200616_H
#define TABULATEDRULES_20200616_H

#include "quadrules/SimplexQuadratureRule.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace tndm {

namespace detail {
template <std::size_t D>
using tabulated_rules_t =
    const std::map<unsigned, std::pair<std::vector<std::array<double, D>>, std::vector<double>>>;
extern tabulated_rules_t<3u> JaskowiecSukumar2020;
extern tabulated_rules_t<2u> WitherdenVincent2015_Tri;
extern tabulated_rules_t<3u> WitherdenVincent2015_Tet;
} // namespace detail

template <class Derived> class TabulatedRules {
public:
    /**
     * @brief Returns size of rule with at least minQuadOrder. Returns numeric_limits max if there
     * is no such rule.
     */
    static std::size_t size(unsigned minQuadOrder) {
        auto rule = Derived::rules().lower_bound(minQuadOrder);
        return rule != Derived::rules().end() ? rule->second.first.size()
                                              : std::numeric_limits<std::size_t>::max();
    }

    static auto get(unsigned minQuadOrder) {
        auto rule = Derived::rules().lower_bound(minQuadOrder);
        if (rule != Derived::rules().end()) {
            SimplexQuadratureRule<Derived::dim()> result(rule->second.first.size(), rule->first);
            result.points() = rule->second.first;
            result.weights() = rule->second.second;
            Derived::transform(result);
            assert(result.points().size() == result.weights().size());
#ifndef NDEBUG
            for (auto& wgt : result.weights()) {
                assert(wgt > 0.0);
            }
#endif
            return result;
        }
        return SimplexQuadratureRule<Derived::dim()>();
    }
};

/**
 * @brief J. Jaśkowiec and N. Sukumar, High-order cubature rules for tetrahedra. Int J Numer Methods
 * Eng. 2020; 121: 2418-2436. https://doi.org/10.1002/nme.6313
 */
class JaskowiecSukumar2020 : public TabulatedRules<JaskowiecSukumar2020> {
public:
    static constexpr std::size_t dim() { return 3u; }
    static constexpr detail::tabulated_rules_t<3u> const& rules() {
        return detail::JaskowiecSukumar2020;
    }
    static void transform(SimplexQuadratureRule<3u>& result) {
        // Divide by volume of reference tetrahedron
        for (auto& wgt : result.weights()) {
            wgt /= 6.0;
        }
    }
};

/**
 * @brief F.D. Witherden and P.E. Vincent, On the identification of symmetric quadrature rules for
 * finite element methods. Comput Math Appl; 69.10: 1232-1241.
 * https://doi.org/10.1016/j.camwa.2015.03.017
 */
template <std::size_t D>
class WitherdenVincent2015 : public TabulatedRules<WitherdenVincent2015<D>> {
public:
    static constexpr std::size_t dim() { return D; }
    static constexpr detail::tabulated_rules_t<D> const& rules() {
        if constexpr (D == 2u) {
            return detail::WitherdenVincent2015_Tri;
        } else if constexpr (D == 3u) {
            return detail::WitherdenVincent2015_Tet;
        } else {
            static_assert("No rules for requested dimension.");
        }
    }
    // Map from {-1,1}^D reference tet to {0,1}^D reference tet
    static void transform(SimplexQuadratureRule<D>& result) {
        for (auto& pt : result.points()) {
            for (auto& x : pt) {
                x = 0.5 * x + 0.5;
            }
        }
        // Multiply weights with 2^-D
        for (auto& wgt : result.weights()) {
            wgt /= (1 << D);
        }
    }
};

} // namespace tndm

#endif // TABULATEDRULES_20200616_H
