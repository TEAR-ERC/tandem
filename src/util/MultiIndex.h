#ifndef MULTIINDEX_H
#define MULTIINDEX_H

#include <array>
#include <cstddef>
#include <cstdint>

namespace tndm {

template <std::size_t D>
uint64_t flatten(std::array<uint64_t, D> const& idx, std::array<uint64_t, D> const& shape) {
    uint64_t flat = 0;
    for (int d = D-1; d >= 0; --d) {
        flat *= shape[d];
        flat += idx[d];
    }
    return flat;
}

template <std::size_t D>
std::array<uint64_t, D> unflatten(uint64_t idx, std::array<uint64_t, D> const& shape) {
    std::array<uint64_t, D> a;
    for (std::size_t d = 0; d < D; ++d) {
        a[d] = idx % shape[d];
        idx /= shape[d];
    }
    return a;
}

template <std::size_t D>
std::array<uint64_t, D> operator+(std::array<uint64_t, D> const& lhs,
                                  std::array<uint64_t, D> const& rhs) {
    std::array<uint64_t, D> result;
    for (std::size_t d = 0; d < D; ++d) {
        result[d] = lhs[d] + rhs[d];
    }
    return result;
}

template <std::size_t D>
std::array<uint64_t, D> operator-(std::array<uint64_t, D> const& lhs,
                                  std::array<uint64_t, D> const& rhs) {
    std::array<uint64_t, D> result;
    for (std::size_t d = 0; d < D; ++d) {
        result[d] = lhs[d] - rhs[d];
    }
    return result;
}
}

#endif // MULTIINDEX_H
