#ifndef TNDM_MATH_H
#define TNDM_MATH_H

#include <cassert>
#include <array>
#include <numeric>

namespace tndm {

constexpr inline unsigned binom(unsigned n, unsigned k) {
    return (k == 0 || k == n) ? 1 :
           (k == 1 || k == n-1) ? n :
           binom(n - 1, k - 1) + binom(n - 1, k);
}

template<unsigned k> class Choose {
public:
    Choose(unsigned n) : n(n) {
        assert(n >= k);
        std::iota(c.begin(), c.end(), 0);
    }

    std::array<unsigned,k> const& current() const {
        return c;
    }

    bool next() {
        for (int i = static_cast<int>(k-1); i >= 0; --i) {
            if (c[i] < n-(k-i)) {
                ++c[i];
                std::iota(c.begin()+(i+1), c.end(), c[i]+1);
                return true;
            }
        }
        return false;
    }
private:
    std::array<unsigned,k> c;
    unsigned n;
};

}

#endif // TNDM_MATH_H
