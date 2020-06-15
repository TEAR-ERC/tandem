#ifndef NME2020_20200615_H
#define NME2020_20200615_H

#include "quadrules/SimplexQuadratureRule.h"

#include <array>
#include <cassert>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace tndm {

namespace detail {
extern const std::map<unsigned, std::pair<std::vector<std::array<double, 3>>, std::vector<double>>>
    NME2020;
}

/**
 * @brief J. Jaśkowiec and N. Sukumar, High-order cubature rules for tetrahedra. Int J Numer Methods
 * Eng. 2020; 121: 2418-2436. https://doi.org/10.1002/nme.6313
 */
class NME2020 {
public:
    /**
     * @brief Returns size of rule with at least minQuadOrder. Returns numeric_limits max if there
     * is no such rule.
     */
    static std::size_t size(unsigned minQuadOrder) {
        auto rule = detail::NME2020.lower_bound(minQuadOrder);
        return rule != detail::NME2020.end() ? rule->second.first.size()
                                             : std::numeric_limits<std::size_t>::max();
    }

    static SimplexQuadratureRule<3u> get(unsigned minQuadOrder) {
        auto rule = detail::NME2020.lower_bound(minQuadOrder);
        if (rule != detail::NME2020.end()) {
            SimplexQuadratureRule<3u> result(rule->second.first.size(), rule->first);
            result.points() = rule->second.first;
            result.weights() = rule->second.second;
            // Divide by volume of reference tetrahedron
            for (auto& wgt : result.weights()) {
                wgt /= 6.0;
            }
            assert(result.points().size() == result.weights().size());
            return result;
        }
        return SimplexQuadratureRule<3u>();
    }
};

} // namespace tndm

#endif // NME2020_20200615_H
