#ifndef GEOMETRY_VECTOR_H
#define GEOMETRY_VECTOR_H

#include <array>
#include <cstddef>

namespace tndm {

template <typename RealT, std::size_t D>
std::array<RealT, D> operator+(std::array<RealT, D> const& lhs, std::array<RealT, D> const& rhs) {
    std::array<RealT, D> result;
    for (std::size_t d = 0; d < D; ++d) {
        result[d] = lhs[d] + rhs[d];
    }
    return result;
}

template <typename RealT, std::size_t D>
std::array<RealT, D> operator-(std::array<RealT, D> const& lhs, std::array<RealT, D> const& rhs) {
    std::array<RealT, D> result;
    for (std::size_t d = 0; d < D; ++d) {
        result[d] = lhs[d] - rhs[d];
    }
    return result;
}

template <typename RealT, std::size_t D>
std::array<RealT, D> operator*(RealT lhs, std::array<RealT, D> const& rhs) {
    std::array<RealT, D> result;
    for (std::size_t d = 0; d < D; ++d) {
        result[d] = lhs * rhs[d];
    }
    return result;
}

template <typename RealT, std::size_t D>
std::array<RealT, D> operator*(std::array<RealT, D> const& lhs, RealT rhs) {
    return rhs * lhs;
}

template <typename RealT, std::size_t D>
RealT dot(std::array<RealT, D> const& lhs, std::array<RealT, D> const& rhs) {
    RealT result = 0.0;
    for (std::size_t d = 0; d < D; ++d) {
        result += lhs[d] * rhs[d];
    }
    return result;
}

template <typename RealT, std::size_t D> RealT norm(std::array<RealT, D> const& x) {
    return sqrt(dot(x, x));
}

template <typename RealT, std::size_t D>
std::array<RealT, D> normalize(std::array<RealT, D> const& x) {
    return x * (1.0 / norm(x));
}

template <typename RealT>
std::array<RealT, 3u> cross(std::array<RealT, 3u> const& lhs, std::array<RealT, 3u> const& rhs) {
    std::array<RealT, 3u> c;
    for (std::size_t d = 0; d < 3u; ++d) {
        c[d] = lhs[(d + 1u) % 3u] * rhs[(d + 2u) % 3u] - lhs[(d + 2u) % 3u] * rhs[(d + 1u) % 3u];
    }
    return c;
}

} // namespace tndm

#endif // GEOMETRY_VECTOR_H
