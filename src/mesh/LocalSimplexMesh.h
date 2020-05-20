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
            g2l[v[0]] = local++;
        }
        setG2L(std::make_index_sequence<D + 1>{});
    }

    template <std::size_t DD> auto faces() const { return std::get<DD>(lfs); }

    auto vertices() const { return faces<0>(); }
    auto edges() const { return faces<1>(); }
    auto facets() const { return faces<D - 1>(); }

    /*template<typename RealT, std::size_t Dout = D>
    std::vector<RealT> flatVertices() const {
        static_assert(Dout >= D);
        std::vector<RealT> vout;
        vout.reserve(Dout*verts.size());
        for (auto& v : verts) {
            std::size_t d;
            for (d = 0; d < D; ++d) {
                vout.push_back(static_cast<RealT>(v[d]));
            }
            for (; d < Dout; ++d) {
                vout.push_back(0.0);
            }
        }
        return vout;
    }

    template<typename IntT>
    std::vector<IntT> flatElements() const {
        std::vector<IntT> eout;
        eout.reserve((D+1)*elems.size());
        for (auto& e : elems) {
            for (auto& p : e) {
                eout.push_back(static_cast<IntT>(p));
            }
        }
        return eout;
    }

    std::size_t numVertices() const {
        return verts.size();
    }

    std::size_t numElements() const {
        return elems.size();
    }*/

private:
    template <std::size_t... Is> void setG2L(std::index_sequence<Is...>) {
        (faces<Is>().setG2L(&g2l), ...);
    }

    storage_t lfs;
    std::unordered_map<int, int> g2l;
};

} // namespace tndm

#endif // LOCALSIMPLEXMESH_H
