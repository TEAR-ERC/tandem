#ifndef LOCALSIMPLEXMESH_H
#define LOCALSIMPLEXMESH_H

#include "LocalFaces.h"
#include "MeshData.h"
#include "Simplex.h"
#include "util/Utility.h"

#include <array>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tndm {

/**
 * @brief Class that holds a distributed D-simplex mesh with ghost entities.
 *
 * @tparam D simplex dimension
 */
template <std::size_t D> class LocalSimplexMesh {
public:
    using storage_t = ntuple_t<LocalFaces, D + 1>;
    template <std::size_t DD>
    using g2l_t = std::unordered_map<Simplex<DD>, std::size_t, SimplexHash<DD>>;

    LocalSimplexMesh(storage_t&& localFaces) : lfs(std::move(localFaces)) {
        makeG2LMaps(std::make_index_sequence<D + 1>());
        makeUpwardMaps(std::make_index_sequence<D>());
    }

    template <std::size_t DD> auto const& faces() const { return std::get<DD>(lfs); }
    template <std::size_t DD> std::size_t size() const { return faces<DD>().size(); }
    template <std::size_t DD> auto const& g2l() const { return std::get<DD>(g2lMaps); }

    auto const& vertices() const { return faces<0>(); }
    std::size_t numVertices() const { return size<0>(); }
    auto const& edges() const { return faces<1>(); }
    std::size_t numEdges() const { return size<1>(); }
    auto const& facets() const { return faces<D - 1>(); }
    std::size_t numFacets() const { return size<D - 1>(); }
    auto const& elements() const { return faces<D>(); }
    std::size_t numElements() const { return size<D>(); }

    template <std::size_t Dto, std::size_t Dfrom> auto downward(std::size_t lid) const {
        return downward<Dto, Dfrom>(faces<Dfrom>()[lid]);
    }

    template <std::size_t Dto, std::size_t Dfrom> auto downward(Simplex<Dfrom> const& face) const {
        auto down = face.template downward<Dto>();
        std::array<std::size_t, down.size()> lids;
        auto it = lids.begin();
        auto& map = g2l<Dto>();
        for (auto& d : down) {
            *(it++) = map.at(d);
        }
        return lids;
    }

    template <std::size_t Dfrom> auto const& upward(std::size_t lid) const {
        static_assert(Dfrom < D);
        assert(lid < size<Dfrom>());
        return upwardMaps[Dfrom].at(lid);
    }

    template <std::size_t Dfrom> auto const& upward(Simplex<Dfrom> const& face) const {
        return upward<Dfrom>(g2l<Dfrom>()[face]);
    }

private:
    using upward_map_t = std::vector<std::vector<std::size_t>>;

    template <std::size_t DD> void makeG2LMap() {
        auto& map = std::get<DD>(g2lMaps);
        map.clear();
        std::size_t local = 0;
        for (auto& f : faces<DD>()) {
            map[f] = local++;
        }
    }
    template <std::size_t... Is> void makeG2LMaps(std::index_sequence<Is...>) {
        (makeG2LMap<Is>(), ...);
    }

    template <std::size_t DD> void makeUpwardMap() {
        auto& map = upwardMaps[DD];
        map.clear();
        map.resize(size<DD>());
        auto& g2lUp = g2l<DD + 1>();
        auto& g2lDown = g2l<DD>();
        for (auto& f : faces<DD + 1>()) {
            std::size_t uLid = g2lUp.at(f);
            for (auto& d : f.downward()) {
                map[g2lDown.at(d)].push_back(uLid);
            }
        }
    }
    template <std::size_t... Is> void makeUpwardMaps(std::index_sequence<Is...>) {
        (makeUpwardMap<Is>(), ...);
    }

    storage_t lfs;
    ntuple_t<g2l_t, D + 1> g2lMaps;
    std::array<upward_map_t, D> upwardMaps;
};

} // namespace tndm

#endif // LOCALSIMPLEXMESH_H
