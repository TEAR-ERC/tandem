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
namespace detail {

template <std::size_t D, std::size_t DD> class BarycentricSolver {
public:
    BarycentricSolver() {}
    BarycentricSolver(std::array<unsigned, DD + 1> const& is,
                      std::array<std::array<double, D + 1>, D + 1> const& dot_products)
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

    std::array<double, DD + 1> solve(std::array<double, D + 1> const& dot_rhs) const {
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

template <std::size_t D, std::size_t... PlexDs> class SimplexDistance {
public:
    SimplexDistance() {}
    SimplexDistance(std::array<std::array<double, D>, D + 1> const& vertices)
        : vertices_(vertices) {
        std::array<std::array<double, D + 1>, D + 1> dot_products;
        for (std::size_t i = 0; i < D + 1; ++i) {
            for (std::size_t j = 0; j < D + 1; ++j) {
                dot_products[i][j] = dot(vertices_[i], vertices_[j]);
            }
        }
        (init_solvers<PlexDs>(dot_products), ...);
    }

    auto closest(std::array<double, D> const& point) const {
        std::array<double, D + 1> dot_rhs;
        for (std::size_t i = 0; i < D + 1; ++i) {
            dot_rhs[i] = dot(point, vertices_[i]);
        }
        auto c2 = closest2<PlexDs...>(point, dot_rhs);
        c2.first = sqrt(c2.first);
        return c2;
    }

private:
    template <std::size_t DD>
    void init_solvers(std::array<std::array<double, D + 1>, D + 1> const& dot_products) {
        Choose<DD + 1> choose(D + 1);
        auto& bary = std::get<DD>(bary_);
        std::size_t f = 0;
        do {
            assert(f < bary.size());
            bary[f++] = BarycentricSolver<D, DD>(choose.current(), dot_products);
        } while (choose.next());
    }

    template <std::size_t DD_head, std::size_t... DD_tail>
    std::pair<double, std::array<double, D>>
    closest2(std::array<double, D> const& point, std::array<double, D + 1> const& dot_rhs) const {
        auto& bary = std::get<DD_head>(bary_);
        auto result = std::make_pair(std::numeric_limits<double>::max(), std::array<double, D>{});
        for (auto const& b : bary) {
            auto lambda = b.solve(dot_rhs);
            if (std::all_of(lambda.begin(), lambda.end(), [](double x) { return x >= 0; })) {
                std::array<double, D> x{};
                for (std::size_t j = 0; j < lambda.size(); ++j) {
                    x = x + lambda[j] * vertices_[b.is()[j]];
                }
                auto diff = x - point;
                double dist2 = dot(diff, diff);
                if (dist2 < result.first) {
                    result.first = dist2;
                    result.second = x;
                }
            }
        }
        if constexpr (sizeof...(DD_tail) > 0) {
            auto other_result = closest2<DD_tail...>(point, dot_rhs);
            if (other_result.first < result.first) {
                return other_result;
            }
        }
        return result;
    }

    std::array<std::array<double, D>, D + 1> vertices_;

    template <std::size_t DD>
    using Solvers = std::array<BarycentricSolver<D, DD>, binom(D + 1, DD + 1)>;
    std::tuple<Solvers<PlexDs>...> bary_;
};

} // namespace detail

template <std::size_t D> class SimplexDistance {
public:
    SimplexDistance(std::array<std::array<double, D>, D + 1> const& vertices) : impl_(vertices) {}

    auto closest(std::array<double, D> const& point) const { return impl_.closest(point); }

private:
    template <std::size_t... PlexDs>
    constexpr static auto impl_type(std::index_sequence<PlexDs...>) {
        return detail::SimplexDistance<D, PlexDs...>{};
    }
    decltype(impl_type(std::make_index_sequence<D + 1>{})) impl_;
};

} // namespace tndm

#endif // SIMPLEXDISTANCE_20210415_H
