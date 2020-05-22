#ifndef SIMPLEX_H
#define SIMPLEX_H

#include <cstddef>
#include <cassert>
#include <array>
#include <algorithm>
#include <utility>

#include "util/Math.h"

namespace tndm {

template<std::size_t D> class SimplexBase : public std::array<int,D+1> {
public:
    using base_t = std::array<int,D+1>;

    SimplexBase() {}

    SimplexBase(std::initializer_list<int> l) {
        assert(l.size() == D+1);
        std::copy(l.begin(), l.end(), base_t::begin());
        std::sort(base_t::begin(), base_t::end());
    }

    SimplexBase(std::array<int,D+1> const& other) : base_t(other) { sort(); }
    SimplexBase(SimplexBase<D> const& other) : base_t(other) { }
    SimplexBase(SimplexBase<D>&& other) : base_t(std::move(other)) { }
    SimplexBase<D>& operator=(SimplexBase<D> const& other) {
        std::copy(other.begin(), other.end(), this->begin());
        return *this;
    }

private:
    void sort() {
        std::sort(base_t::begin(), base_t::end());
    }
};

template <std::size_t DD, std::size_t D> struct Downward;

template<std::size_t D> class Simplex : public SimplexBase<D> {
public:
    using SimplexBase<D>::SimplexBase;

    template<std::size_t DD = D-1>
    auto downward() const {
        Downward<DD, D> down;
        return down(*this);
    }
};

template <> class Simplex<0> : public SimplexBase<0> {};

template <std::size_t DD, std::size_t D> struct Downward {
    auto operator()(Simplex<D> const& plex) {
        static_assert(0 <= DD);
        static_assert(DD < D);

        // Choose k out of n vertices
        constexpr int n = D+1;
        constexpr int k = DD+1;
        std::array<Simplex<DD>,binom(n,k)> dws;
        auto i = dws.begin();
        
        Choose<k> choose(n);
        do {
            auto j = i->begin();
            for (auto& c : choose.current()) {
                *(j++) = plex[c];
            }
            ++i;
        } while (choose.next());

        return dws;
    }
};

template <std::size_t D> struct Downward<0, D> {
    auto operator()(Simplex<D> const& plex) {
        static_assert(0 < D);

        std::array<Simplex<0>, D + 1> dws;
        for (std::size_t d = 0; d < D + 1; ++d) {
            dws[d][0] = plex[d];
        }
        return dws;
    }
};

template<std::size_t D> struct SimplexHash {
    std::size_t operator()(Simplex<D> const& plex) const noexcept {
        std::hash<typename Simplex<D>::value_type> hasher;
        std::size_t hash = hasher(plex[0]);
        // From boost::hash_combine
        for (auto it = plex.begin()+1; it != plex.end(); ++it) {
            hash ^= hasher(*it) + 0x9e3779b9 + ((*it) << 6) + ((*it) >> 2);
        }
        return hash;
    }
};

}


#endif // SIMPLEX_H
