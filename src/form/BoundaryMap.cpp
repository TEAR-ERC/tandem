#include "BoundaryMap.h"
#include "form/DGOperatorTopo.h"
#include "form/FacetInfo.h"
#include "mesh/LocalSimplexMesh.h"

#include <cassert>
#include <limits>

namespace tndm {

template <std::size_t D>
BoundaryMap::BoundaryMap(LocalSimplexMesh<D> const& mesh, BC bc, MPI_Comm comm) {
    auto const& elements = mesh.elements();
    auto const& facets = mesh.facets();
    auto boundaryData = dynamic_cast<BoundaryData const*>(facets.data());
    if (!boundaryData) {
        throw std::runtime_error("Boundary conditions not set.");
    }
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::size_t numLocalFacets = mesh.facets().localSize();
    std::vector<std::pair<int, std::size_t>> theFctNos;
    theFctNos.reserve(numLocalFacets);

    for (std::size_t fctNo = 0; fctNo < numLocalFacets; ++fctNo) {
        if (boundaryData->getBoundaryConditions()[fctNo] == bc) {
            auto elNos = mesh.template upward<D - 1u>(fctNo);
            assert(elNos.size() >= 1u && elNos.size() <= 2u);
            int other_rank = -1;
            for (int i = 0; i < elNos.size(); ++i) {
                if (elements.owner(elNos[i]) == rank) {
                    other_rank = elements.owner(elNos[(i + 1) % elNos.size()]);
                }
            }
            if (other_rank < 0) {
                throw std::logic_error("BoundaryMap: That was not supposed to happen.");
            }
            theFctNos.push_back(std::make_pair(other_rank, fctNo));
        }
    }

    /* Sort by
     * 1) Faces owned by this rank
     * 2) Rank number
     * 3) GID
     */
    std::sort(theFctNos.begin(), theFctNos.end(), [&rank, &facets](auto const& a, auto const& b) {
        return std::make_tuple(!(rank <= a.first), a.first, facets.l2cg(a.second)) <
               std::make_tuple(!(rank <= b.first), b.first, facets.l2cg(b.second));
    });

    bndNos_.resize(numLocalFacets, std::numeric_limits<std::size_t>::max());
    fctNos_.reserve(theFctNos.size());
    local_size_ = 0;
    std::unordered_map<int, std::vector<std::size_t>> send_map;
    std::unordered_map<int, std::vector<std::size_t>> recv_map;
    for (auto const& [other_rank, fctNo] : theFctNos) {
        std::size_t bndNo = fctNos_.size();
        bndNos_[fctNo] = bndNo;
        fctNos_.push_back(fctNo);
        if (rank <= other_rank) {
            send_map[other_rank].push_back(bndNo);
            ++local_size_;
            assert(fctNos_.size() == local_size_);
        } else {
            recv_map[other_rank].push_back(bndNo);
        }
    }

    scatter_plan_ = std::make_shared<ScatterPlan>(send_map, recv_map);
}

template BoundaryMap::BoundaryMap(LocalSimplexMesh<1> const&, BC, MPI_Comm);
template BoundaryMap::BoundaryMap(LocalSimplexMesh<2> const&, BC, MPI_Comm);
template BoundaryMap::BoundaryMap(LocalSimplexMesh<3> const&, BC, MPI_Comm);

} // namespace tndm
