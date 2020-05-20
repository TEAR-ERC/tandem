#ifndef LOCALSIMPLEXMESH_H
#define LOCALSIMPLEXMESH_H

#include "LocalFaces.h"
#include "Simplex.h"

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace tndm {

namespace detail {

template <std::size_t... Is> auto lfTuple(std::index_sequence<Is...>) {
    return std::tuple<LocalFaces<Is>...>{};
}
template <std::size_t D> auto lfTuple() { return lfTuple(std::make_index_sequence<D>{}); }
} // namespace detail

template<std::size_t D>
class LocalSimplexMesh {
public:
    using storage_t = decltype(detail::lfTuple<D + 1>());

    LocalSimplexMesh(storage_t&& localFaces) : lfs(std::move(localFaces)) {
        int local = 0;
        for (auto& v : vertices()) {
            g2l_[v[0]] = local++;
        }
    }

    template <std::size_t DD> auto const& faces() const { return std::get<DD>(lfs); }

    auto vertices() const { return faces<0>(); }
    auto edges() const { return faces<1>(); }
    auto facets() const { return faces<D - 1>(); }
    auto elements() const { return faces<D>(); }

    int g2l(int gid) const { return g2l_.at(gid); }

private:
    storage_t lfs;
    std::unordered_map<int, int> g2l_;
};

} // namespace tndm

#endif // LOCALSIMPLEXMESH_H
