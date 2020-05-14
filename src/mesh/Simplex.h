#ifndef SIMPLEX_H
#define SIMPLEX_H

#include <cstddef>
#include <cassert>
#include <array>
#include <algorithm>

namespace tndm {

template<std::size_t D>
class Simplex {
public:
    Simplex() {}

    Simplex(std::initializer_list<int> l) {
        assert(l.size() == D+1);
        std::copy(l.begin(), l.end(), points.begin());
        std::sort(points.begin(), points.end());
    }

    Simplex(std::array<int,D+1> const& other) {
        points = other;
        std::sort(points.begin(), points.end());
    }

    auto downward() {
        std::array<Simplex<D-1>,D+1> dws;
        for (std::size_t f = 0; f < D+1; ++f) {
            dws[f] = downward(f);
        }
        return dws;
    }

    Simplex<D-1> downward(int f) {
        Simplex<D-1> dw;
        auto out = dw.points.begin();
        for (std::size_t j = 0; j < D+1; ++j) {
            if (j == f) {
                continue;
            }
            *(out++) = points[j];
        }
        return dw;
    }

    auto begin() const noexcept { return points.cbegin(); }
    auto end() const noexcept { return points.cend(); }

private:
    friend class Simplex<D+1>;
    std::array<int,D+1> points;
};

template<> struct Simplex<0> {};

}


#endif // SIMPLEX_H
