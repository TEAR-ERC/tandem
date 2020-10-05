#ifndef DGOPERATORTOPO_20200909_H
#define DGOPERATORTOPO_20200909_H

#include "form/FacetInfo.h"
#include "mesh/LocalSimplexMesh.h"
#include "parallel/Scatter.h"

#include "mneme/storage.hpp"
#include <mpi.h>

#include <cstddef>
#include <vector>

namespace tndm {

class DGOperatorTopo {
public:
    template <std::size_t D> DGOperatorTopo(LocalSimplexMesh<D> const& mesh, MPI_Comm comm);
    virtual ~DGOperatorTopo() = default;

    std::size_t numElements() const { return numElems_; }
    std::size_t numLocalElements() const { return numLocalElems_; }
    std::size_t numLocalFacets() const { return numLocalFacets_; }
    Scatter& elementScatter() { return elementScatter_; }

    FacetInfo const& info(std::size_t fctNo) const { return fctInfo[fctNo]; }
    std::size_t gid(std::size_t elNo) const { return volInfo[elNo].template get<GID>(); }
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

    MPI_Comm comm() const { return comm_; }

protected:
    std::size_t numElems_;
    std::size_t numLocalElems_;
    std::size_t numLocalFacets_;
    Scatter elementScatter_;
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
};

} // namespace tndm

#endif // DGOPERATORTOPO_20200909_H
