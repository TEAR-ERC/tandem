#ifndef COMBINATORICS_H
#define COMBINATORICS_H

#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <numeric>

namespace tndm {

/**V
 * @brief Returns n choose k.
 */
constexpr inline std::size_t binom(unsigned n, unsigned k) {
    return (k == 0 || k == n) ? 1
                              : (k == 1 || k == n - 1) ? n : binom(n - 1, k - 1) + binom(n - 1, k);
}

template <unsigned k> class Choose {
public:
    Choose(unsigned n) : n(n) {
        assert(n >= k);
        std::iota(c.begin(), c.end(), 0);
    }

    std::array<unsigned, k> const& current() const { return c; }

    bool next() {
        for (int i = static_cast<int>(k - 1); i >= 0; --i) {
            if (c[i] < n - (k - i)) {
                ++c[i];
                std::iota(c.begin() + (i + 1), c.end(), c[i] + 1);
                return true;
            }
        }
        return false;
    }

private:
    std::array<unsigned, k> c;
    unsigned n;
};

template <unsigned D, bool EnableLess> class AllIntegerSumsIterator {
public:
    static_assert(D > 0);

    using iterator_category = std::forward_iterator_tag;
    using value_type = std::array<unsigned, D>;
    using reference = value_type&;
    using marker_type = std::array<unsigned, D>;

    AllIntegerSumsIterator(unsigned f, unsigned maxSum) {
        std::fill(markers.begin(), markers.end(), f);
        if (!EnableLess) {
            markers.back() = maxSum;
        }
    }

    bool operator!=(AllIntegerSumsIterator const& other) const { return markers != other.markers; }
    bool operator==(AllIntegerSumsIterator const& other) const { return markers == other.markers; }

    auto& operator++() {
        unsigned d = 0;
        while (d < D - 1 && markers[d] == markers[d + 1]) {
            ++d;
        }
        ++markers[d];
        std::fill(markers.begin(), markers.begin() + d, 0);
        return *this;
    }
    auto operator++(int) {
        AllIntegerSumsIterator copy(*this);
        ++(*this);
        return copy;
    }

    value_type operator*() {
        value_type i;
        std::adjacent_difference(markers.begin(), markers.end(), i.rbegin());
        return i;
    }

private:
    marker_type markers;
};

/**
 * @brief Create all integer sums.
 *
 * Can be used to iterate over all integer D-tuples (x_1,...,x_D) such that
 *
 * x_1 + ... + x_D <= maxSum
 *
 * if EnableLess is false, then
 *
 * x_1 + ... + x_D = maxSum
 *
 */
template <unsigned D, bool EnableLess = true> class AllIntegerSums {
public:
    AllIntegerSums(unsigned maxSum) : maxSum(maxSum) {}

    auto begin() { return AllIntegerSumsIterator<D, EnableLess>(0, maxSum); }
    auto end() { return ++AllIntegerSumsIterator<D, EnableLess>(maxSum, maxSum); }

private:
    unsigned maxSum;
};

} // namespace tndm

#endif // COMBINATORICS_H
