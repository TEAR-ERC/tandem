#ifndef SIMPLEXDISTANCE_20210415_H
#define SIMPLEXDISTANCE_20210415_H

#include "geometry/Vector.h"
#include "util/Combinatorics.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace tndm {

template <std::size_t D, std::size_t V> struct SimplexDistanceResult {
    double dist;
    std::array<double, D> xi;
    std::array<double, V> bary;
};

namespace detail {

template <std::size_t D, std::size_t V, std::size_t DD> class BarycentricSolver {
public:
    BarycentricSolver() {}
    BarycentricSolver(std::array<unsigned, DD + 1> const& is,
                      std::array<std::array<double, V>, V> const& dot_products)
        : is_(is) {
        Matrix A;
        for (std::size_t j = 0; j < DD + 1; ++j) {
            for (std::size_t i = 0; i < DD; ++i) {
                A(i, j) = dot_products[is_[i + 1]][is_[j]] - dot_products[is_[0]][is_[j]];
            }
            A(DD, j) = 1.0;
        }
        lu_ = A.partialPivLu();
    }

    std::array<double, DD + 1> solve(std::array<double, V> const& dot_rhs) const {
        Vector b;
        for (std::size_t i = 0; i < DD; ++i) {
            b(i) = dot_rhs[is_[i + 1]] - dot_rhs[is_[0]];
        }
        b(DD) = 1.0;
        std::array<double, DD + 1> lambda;
        Eigen::Map<Vector>(lambda.data()) = lu_.solve(b);
        return lambda;
    }

    auto const& is() const { return is_; }

private:
    using Matrix = Eigen::Matrix<double, DD + 1, DD + 1>;
    using Vector = Eigen::Matrix<double, DD + 1, 1>;

    std::array<unsigned, DD + 1> is_;
    Eigen::PartialPivLU<Matrix> lu_;
};

template <std::size_t D, std::size_t V, std::size_t... PlexDs> class SimplexDistance {
public:
    SimplexDistance() {}
    SimplexDistance(std::array<std::array<double, D>, V> const& vertices) : vertices_(vertices) {
        std::array<std::array<double, V>, V> dot_products;
        for (std::size_t i = 0; i < V; ++i) {
            for (std::size_t j = 0; j < V; ++j) {
                dot_products[i][j] = dot(vertices_[i], vertices_[j]);
            }
        }
        (init_solvers<PlexDs>(dot_products), ...);
    }

    auto closest(std::array<double, D> const& point) const {
        std::array<double, V> dot_rhs;
        for (std::size_t i = 0; i < V; ++i) {
            dot_rhs[i] = dot(point, vertices_[i]);
        }
        auto c2 = closest2<PlexDs...>(point, dot_rhs);
        c2.dist = sqrt(c2.dist);
        return c2;
    }

private:
    template <std::size_t DD>
    void init_solvers(std::array<std::array<double, V>, V> const& dot_products) {
        Choose<DD + 1> choose(V);
        auto& bary = std::get<DD>(bary_);
        std::size_t f = 0;
        do {
            assert(f < bary.size());
            bary[f++] = BarycentricSolver<D, V, DD>(choose.current(), dot_products);
        } while (choose.next());
    }

    template <std::size_t DD_head, std::size_t... DD_tail>
    SimplexDistanceResult<D, V> closest2(std::array<double, D> const& point,
                                         std::array<double, V> const& dot_rhs) const {
        auto& bary = std::get<DD_head>(bary_);
        auto result = SimplexDistanceResult<D, V>{std::numeric_limits<double>::max(), {}, {}};
        for (auto const& b : bary) {
            auto lambda = b.solve(dot_rhs);
            if (std::all_of(lambda.begin(), lambda.end(), [](double x) { return x >= 0; })) {
                std::array<double, D> x{};
                for (std::size_t j = 0; j < lambda.size(); ++j) {
                    x = x + lambda[j] * vertices_[b.is()[j]];
                }
                auto diff = x - point;
                double dist2 = dot(diff, diff);
                if (dist2 < result.dist) {
                    result.dist = dist2;
                    result.xi = x;
                    result.bary.fill(0.0);
                    for (std::size_t i = 0; i < b.is().size(); ++i) {
                        result.bary[b.is()[i]] = lambda[i];
                    }
                }
            }
        }
        if constexpr (sizeof...(DD_tail) > 0) {
            auto other_result = closest2<DD_tail...>(point, dot_rhs);
            if (other_result.dist < result.dist) {
                return other_result;
            }
        }
        return result;
    }

    std::array<std::array<double, D>, V> vertices_;

    template <std::size_t DD>
    using Solvers = std::array<BarycentricSolver<D, V, DD>, binom(V, DD + 1)>;
    std::tuple<Solvers<PlexDs>...> bary_;
};

} // namespace detail

/**
 * @brief Class to find closest point in simplex
 *
 * @tparam D space dimension
 * @tparam V number of vertices of simplex (1 = point, 2 = line, 3 = triangle, 4 = tet)
 */
template <std::size_t D, std::size_t V> class SimplexDistance {
public:
    SimplexDistance() {}
    SimplexDistance(std::array<std::array<double, D>, V> const& vertices) : impl_(vertices) {}

    auto closest(std::array<double, D> const& point) const { return impl_.closest(point); }

private:
    template <std::size_t... PlexDs>
    constexpr static auto impl_type(std::index_sequence<PlexDs...>) {
        return detail::SimplexDistance<D, V, PlexDs...>{};
    }
    decltype(impl_type(std::make_index_sequence<V>{})) impl_;
};

} // namespace tndm

#endif // SIMPLEXDISTANCE_20210415_H
