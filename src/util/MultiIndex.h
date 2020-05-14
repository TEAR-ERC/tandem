#ifndef MULTIINDEX_H
#define MULTIINDEX_H

#include <cstddef>
#include <array>

namespace tndm {

template<std::size_t D>
int flatten(std::array<int,D> const& idx, std::array<int,D> const& shape) {
    int flat = 0;
    for (int d = D-1; d >= 0; --d) {
        flat *= shape[d];
        flat += idx[d];
    }
    return flat;
}

template<std::size_t D>
std::array<int,D> unflatten(int idx, std::array<int,D> const& shape) {
    std::array<int,D> a;
    for (std::size_t d = 0; d < D; ++d) {
        a[d] = idx % shape[d];
        idx /= shape[d];
    }
    return a;
}

template<std::size_t D>
std::array<int,D> operator+(std::array<int,D> const& lhs, std::array<int,D> const& rhs) {
    std::array<int,D> result;
    for (std::size_t d = 0; d < D; ++d) {
        result[d] = lhs[d] + rhs[d];
    }
    return result;
}

template<std::size_t D>
std::array<int,D> operator-(std::array<int,D> const& lhs, std::array<int,D> const& rhs) {
    std::array<int,D> result;
    for (std::size_t d = 0; d < D; ++d) {
        result[d] = lhs[d] - rhs[d];
    }
    return result;
}

}

#endif // MULTIINDEX_H
