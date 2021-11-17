#ifndef PERMUTATION_20211117_H
#define PERMUTATION_20211117_H

#include <array>
#include <bitset>
#include <cstddef>

namespace tndm {

/**
 * @brief Encode permutation in Lehmer code
 *
 * see Section 5.2 in
 * https://www.cs.helsinki.fi/u/bmmalone/heuristic-search-fall-2013/Korf2008.pdf
 *
 * @tparam N Length of permutation
 * @param p Permutation
 *
 * @return Lehmer code of p
 */
template <std::size_t N>
auto permutation_to_Lehmer(std::array<unsigned, N> const& p) -> std::array<unsigned, N> {
    std::array<unsigned, N> L;
    auto seen = std::bitset<N>{};
    auto l = L.begin();
    for (auto k : p) {
        seen.set(k);
        *l++ = k - (seen << (N - k)).count();
    }
    return L;
}

/**
 * @brief Map Lehmer code to integer in range 0,...,N!-1
 *
 * @tparam N Length of Lehmer code
 * @param L Lehmer code
 *
 * @return Unique integer associated with Lehmer code
 */
template <std::size_t N> auto Lehmer_to_integer(std::array<unsigned, N> const& L) -> unsigned {
    unsigned a = 0;
    unsigned f = 1;
    unsigned i = 1;
    for (auto l = L.rbegin(); l != L.rend(); ++l) {
        a += *l * f;
        f *= i++;
    }
    return a;
}

} // namespace tndm

#endif // PERMUTATION_20211117_H
