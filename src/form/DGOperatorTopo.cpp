#include "DGOperatorTopo.h"
#include "mesh/MeshData.h"

#include <array>
#include <cassert>
#include <iterator>
#include <memory>
#include <stdexcept>

namespace tndm {

template <std::size_t D>
DGOperatorTopo::DGOperatorTopo(LocalSimplexMesh<D> const& mesh, MPI_Comm comm)
    : numElems_(mesh.numElements()), numLocalElems_(mesh.elements().localSize()),
      numLocalFacets_(mesh.facets().localSize()),
      elementScatterPlan_(std::make_shared<ScatterPlan>(mesh.elements(), comm)), comm_(comm),
      fctInfo(mesh.facets().localSize()), volInfo(mesh.numElements()) {
    int rank;
    MPI_Comm_rank(comm_, &rank);

    auto boundaryData = dynamic_cast<BoundaryData const*>(mesh.facets().data());
    if (!boundaryData) {
        throw std::runtime_error("Boundary conditions not set.");
    }

    numInteriorElems_ = 0;
    std::size_t lastElNoInInterior = 0;
    const auto& elements = mesh.elements();
    for (std::size_t elNo = 0; elNo < elements.size(); ++elNo) {
        auto owner = elements.owner(elNo);
        if (owner == rank && elements.getSharedRanks(elNo).size() == 0) {
            ++numInteriorElems_;
            lastElNoInInterior = elNo;
        }
    }
    if (numInteriorElems_ > 0 && lastElNoInInterior + 1 != numInteriorElems_) {
        throw std::runtime_error("Assertion failed. Interior elements are not compact.");
    }

    constexpr auto num_facets = D + 1;
    auto neigh_storage = std::make_shared<neighbour_t>(numLocalElems_ * num_facets);
    neighbourInfo.setStorage(neigh_storage, 0, numLocalElems_, num_facets);

    #pragma omp parallel
    {
        #pragma omp for
        for (std::size_t fctNo = 0; fctNo < numLocalFacets(); ++fctNo) {
            FacetInfo& info = fctInfo[fctNo];

            auto elNos = mesh.template upward<D - 1u>(fctNo);
            assert(elNos.size() >= 1u && elNos.size() <= 2u);
            auto dws = mesh.template downward<D - 1u, D>(elNos[0]);
            auto localFctNo = std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fctNo));
            assert(localFctNo < D + 1u);

            info.up[0] = elNos[0];
            info.g_up[0] = mesh.elements().l2cg(elNos[0]);
            info.localNo[0] = localFctNo;
            info.inside[0] = elNos[0] < numLocalElems_;
            info.bc = boundaryData->getBoundaryConditions()[fctNo];

            if (elNos.size() > 1) {
                auto dwsOther = mesh.template downward<D - 1u, D>(elNos[1]);
                auto localFNoOther = std::distance(
                    dwsOther.begin(), std::find(dwsOther.begin(), dwsOther.end(), fctNo));
                assert(localFNoOther < D + 1u);

                info.up[1] = elNos[1];
                info.g_up[1] = mesh.elements().l2cg(elNos[1]);
                info.localNo[1] = localFNoOther;
                info.inside[1] = elNos[1] < numLocalElems_;
            } else {
                info.up[1] = info.up[0];
                info.g_up[1] = info.g_up[0];
                info.localNo[1] = info.localNo[0];
                info.inside[1] = info.inside[0];
            }
            assert(info.inside[0] || info.inside[1]);

            for (int i = 0; i < 2; ++i) {
                if (info.inside[i]) {
                    auto& si = neighbourInfo[info.up[i]];
                    auto l = info.localNo[i];
                    si[l].fctNo = fctNo;
                    si[l].side = i;
                    si[l].lid = info.up[(i + 1) % 2];
                    si[l].localNo = info.localNo[(i + 1) % 2];
                    si[l].bc = info.bc;
                }
            }
        }

        #pragma omp for
        for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
            volInfo[elNo].template get<GID>() = mesh.elements().l2cg(elNo);
        }

        #pragma omp for
        for (std::size_t elNo = 0; elNo < numLocalElements(); ++elNo) {
            auto dws = mesh.template downward<D - 1u, D>(elNo);
            unsigned numLocal = 1, numGhost = 0;
            for (auto&& dw : dws) {
                auto up = mesh.template upward<D - 1u>(dw);
                for (auto&& u : up) {
                    if (u != elNo) {
                        if (mesh.elements().owner(u) == rank) {
                            ++numLocal;
                        } else {
                            ++numGhost;
                        }
                    }
                }
            }
            // At most D + 1 neighbours plus element itself
            assert(numLocal + numGhost <= D + 2u);
            volInfo[elNo].template get<NumLocalNeighbours>() = numLocal;
            volInfo[elNo].template get<NumGhostNeighbours>() = numGhost;
        }
    }
}

template DGOperatorTopo::DGOperatorTopo(LocalSimplexMesh<2u> const&, MPI_Comm);
template DGOperatorTopo::DGOperatorTopo(LocalSimplexMesh<3u> const&, MPI_Comm);

} // namespace tndm
