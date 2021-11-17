#ifndef UTIL_20211117_H
#define UTIL_20211117_H

#include "geometry/Vector.h"
#include "mesh/Simplex.h"

#include <array>
#include <cstddef>
#include <iterator>
#include <optional>
#include <vector>

namespace tndm {

/**
 * @brief Returns the permutation of the nodes for a permutation of the vertices
 *
 * Let a permutation of the vertices be given by p_i, i=0,...,D, where p_i in {0,...,D}.
 * The affine mapping from the original reference simplex to the permuted reference simplex is
 *
 * eta(xi) = (1 - sum_{i=0}^D xi_i) r_{p_0} + sum_i xi_r r_{p_i},
 *
 * such that
 *
 * eta(xi_i) = r_{p_i}
 *
 * Then we want to find a permutation P_i, i=0,...,B-1, B being the number of nodes,
 * such that we have
 *
 * xi_{P_i} = eta(xi_i).
 *
 * If such a permutation does not exist, std::nullopt is returned.
 *
 * @tparam D Simplex dimension
 * @param degree Polynomial degree
 * @param p Permutation of the vertices
 *
 * @return Node permutation if it exists, std::nullopt otherwise.
 */
template <std::size_t D>
std::optional<std::vector<unsigned>>
node_permutation(std::vector<std::array<double, D>> const& nodes,
                 std::array<unsigned, D + 1> const& p, double tol = 1e-15) {
    auto r = Simplex<D>::referenceSimplexVertices();
    auto eta = [&r, &p](std::array<double, D> const& xi) {
        double sum_xi = std::accumulate(xi.begin(), xi.end(), 0.0);
        auto eta = (1.0 - sum_xi) * r[p[0]];
        for (std::size_t d = 0; d < xi.size(); ++d) {
            eta = eta + xi[d] * r[p[d + 1]];
        }
        return eta;
    };

    auto check = std::vector<unsigned>(nodes.size(), 0);
    auto result = std::vector<unsigned>{};
    result.reserve(nodes.size());
    auto first = nodes.cbegin();
    auto last = nodes.cend();
    for (auto const& xi : nodes) {
        auto eta_xi = eta(xi);
        auto it = first;
        while (it != last && norm(eta_xi - *it) > tol) {
            ++it;
        }
        if (it == last) {
            return std::nullopt;
        }
        auto pos = std::distance(first, it);
        result.emplace_back(pos);
        check[pos] += 1;
    }
    for (auto& c : check) {
        if (c != 1) {
            return std::nullopt;
        }
    }
    return std::make_optional(std::move(result));
}

/**
 * @brief Converts a number of nodes on the D-simplex in the respective polynomial degree
 *
 * @param num_nodes Number of nodes
 *
 * @return Polynomial degree, std::nullopt if number of nodes is incomplete
 */
template <std::size_t D>
auto num_nodes_to_degree(std::size_t num_nodes) -> std::optional<unsigned> {
    unsigned N = 1;
    while (num_nodes > binom(N + D, D)) {
        ++N;
    }
    if (num_nodes != binom(N + D, D)) {
        return std::nullopt;
    }
    return std::make_optional(N);
}

} // namespace tndm

#endif // UTIL_20211117_H
