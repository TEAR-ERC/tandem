#ifndef DGOPERATORTOPO_20200909_H
#define DGOPERATORTOPO_20200909_H

#include "form/FacetInfo.h"
#include "mesh/LocalSimplexMesh.h"
#include "parallel/ScatterPlan.h"

#include "mneme/span.hpp"
#include "mneme/storage.hpp"
#include "mneme/view.hpp"
#include <mpi.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

class DGOperatorTopo {
public:
    template <std::size_t D> DGOperatorTopo(LocalSimplexMesh<D> const& mesh, MPI_Comm comm);
    virtual ~DGOperatorTopo() = default;

    /**
     * @brief Number of elements in interior layer.
     */
    std::size_t numInteriorElements() const { return numInteriorElems_; }
    /**
     * @brief Number of elements owned by the partition, i.e. interior + copy.
     */
    std::size_t numLocalElements() const { return numLocalElems_; }
    /**
     * @brief Number of elements in the partition, i.e. interior + copy + ghost.
     *
     * Interior ids: [0, numInteriorElements).
     * Copy ids:     [numInteriorElements, numLocalElements).
     * Ghost ids:    [numLocalElements, numElements).
     */
    std::size_t numElements() const { return numElems_; }
    std::size_t numLocalFacets() const { return numLocalFacets_; }
    std::shared_ptr<ScatterPlan> elementScatterPlan() { return elementScatterPlan_; }

    FacetInfo const& info(std::size_t fctNo) const { return fctInfo[fctNo]; }
    std::size_t gid(std::size_t elNo) const { return volInfo[elNo].template get<GID>(); }
    std::size_t const* gids() const { return &volInfo[0].template get<GID>(); }
    unsigned numLocalNeighbours(std::size_t elNo) const {
        return volInfo[elNo].template get<NumLocalNeighbours>();
    }
    unsigned const* numLocalNeighbours() const {
        return &volInfo[0].template get<NumLocalNeighbours>();
    }
    unsigned numGhostNeighbours(std::size_t elNo) const {
        return volInfo[elNo].template get<NumGhostNeighbours>();
    }
    unsigned const* numGhostNeighbours() const {
        return &volInfo[0].template get<NumGhostNeighbours>();
    }
    mneme::span<SideInfo> neighbours(std::size_t elNo) const { return neighbourInfo[elNo]; }

    MPI_Comm comm() const { return comm_; }

protected:
    std::size_t numElems_;
    std::size_t numInteriorElems_;
    std::size_t numLocalElems_;
    std::size_t numLocalFacets_;
    std::shared_ptr<ScatterPlan> elementScatterPlan_;
    MPI_Comm comm_;

    std::vector<FacetInfo> fctInfo;

    struct GID {
        using type = std::size_t;
    };
    struct NumLocalNeighbours {
        using type = unsigned;
    };
    struct NumGhostNeighbours {
        using type = unsigned;
    };
    mneme::MultiStorage<mneme::DataLayout::SoA, GID, NumLocalNeighbours, NumGhostNeighbours>
        volInfo;

    struct Side {
        using type = SideInfo;
    };
    using neighbour_t = mneme::SingleStorage<Side>;
    mneme::StridedView<neighbour_t> neighbourInfo;
};

} // namespace tndm

#endif // DGOPERATORTOPO_20200909_H
