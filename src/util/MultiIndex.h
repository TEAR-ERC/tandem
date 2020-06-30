#ifndef MULTIINDEX_H
#define MULTIINDEX_H

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace tndm {

template <std::size_t D, typename T>
T flatten(std::array<T, D> const& idx, std::array<T, D> const& shape) {
    T flat = 0;
    for (std::ptrdiff_t d = D - 1; d >= 0; --d) {
        flat *= shape[d];
        flat += idx[d];
    }
    return flat;
}

template <std::size_t D, typename T>
std::array<T, D> unflatten(T idx, std::array<T, D> const& shape) {
    std::array<T, D> a;
    for (std::size_t d = 0; d < D; ++d) {
        a[d] = idx % shape[d];
        idx /= shape[d];
    }
    return a;
}

template <typename T, typename... Entries>
constexpr std::array<T, sizeof...(Entries)> make_index(Entries&&... entries) {
    return {static_cast<T>(entries)...};
}

template <std::size_t D, typename T, typename U>
std::array<U, D> permute(std::array<T, D> const& permutation, std::array<U, D> const& index) {
    std::array<U, D> a;
    for (std::size_t d = 0; d < D; ++d) {
        assert(permutation[d] < D);
        a[d] = index[permutation[d]];
    }
    return a;
}

template <std::size_t D, typename T>
std::array<T, D> operator+(std::array<T, D> const& lhs, std::array<T, D> const& rhs) {
    std::array<uint64_t, D> result;
    for (std::size_t d = 0; d < D; ++d) {
        result[d] = lhs[d] + rhs[d];
    }
    return result;
}

template <std::size_t D, typename T>
std::array<T, D> operator-(std::array<T, D> const& lhs, std::array<T, D> const& rhs) {
    std::array<T, D> result;
    for (std::size_t d = 0; d < D; ++d) {
        result[d] = lhs[d] - rhs[d];
    }
    return result;
}
} // namespace tndm

#endif // MULTIINDEX_H
