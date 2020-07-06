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

    LocalSimplexMesh(storage_t&& localFaces) : lfs(std::move(localFaces)) {
        makeUpwardMaps(std::make_index_sequence<D>());
    }

    template <std::size_t DD> auto const& faces() const { return std::get<DD>(lfs); }
    template <std::size_t DD> std::size_t size() const { return faces<DD>().size(); }

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
        auto& map = faces<Dto>().g2l();
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
        return upward<Dfrom>(faces<Dfrom>().g2l()[face]);
    }

private:
    using upward_map_t = std::vector<std::vector<std::size_t>>;

    template <std::size_t DD> void makeUpwardMap() {
        auto& map = upwardMaps[DD];
        map.clear();
        map.resize(size<DD>());
        auto& g2lUp = faces<DD + 1>().g2l();
        auto& g2lDown = faces<DD>().g2l();
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
    std::array<upward_map_t, D> upwardMaps;
};

} // namespace tndm

#endif // LOCALSIMPLEXMESH_H
