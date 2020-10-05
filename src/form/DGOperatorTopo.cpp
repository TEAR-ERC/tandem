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
      numLocalFacets_(mesh.facets().localSize()), elementScatter_(mesh.elements(), comm),
      comm_(comm), fctInfo(mesh.facets().localSize()), volInfo(mesh.numElements()) {
    int rank;
    MPI_Comm_rank(comm_, &rank);

    auto boundaryData = dynamic_cast<BoundaryData const*>(mesh.facets().data());
    if (!boundaryData) {
        throw std::runtime_error("Boundary conditions not set.");
    }

#pragma omp parallel
    {
#pragma omp for
        for (std::size_t fctNo = 0; fctNo < numLocalFacets(); ++fctNo) {
            auto elNos = mesh.template upward<D - 1u>(fctNo);
            assert(elNos.size() >= 1u && elNos.size() <= 2u);
            auto dws = mesh.template downward<D - 1u, D>(elNos[0]);
            auto localFctNo = std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fctNo));
            assert(localFctNo < D + 1u);

            fctInfo[fctNo].up[0] = elNos[0];
            fctInfo[fctNo].g_up[0] = mesh.elements().l2cg(elNos[0]);
            fctInfo[fctNo].localNo[0] = localFctNo;
            fctInfo[fctNo].inside[0] = elNos[0] < numLocalElems_;
            fctInfo[fctNo].bc = boundaryData->getBoundaryConditions()[fctNo];

            if (elNos.size() > 1) {
                auto dwsOther = mesh.template downward<D - 1u, D>(elNos[1]);
                auto localFNoOther = std::distance(
                    dwsOther.begin(), std::find(dwsOther.begin(), dwsOther.end(), fctNo));
                assert(localFNoOther < D + 1u);

                fctInfo[fctNo].up[1] = elNos[1];
                fctInfo[fctNo].g_up[1] = mesh.elements().l2cg(elNos[1]);
                fctInfo[fctNo].localNo[1] = localFNoOther;
                fctInfo[fctNo].inside[1] = elNos[1] < numLocalElems_;
            } else {
                fctInfo[fctNo].up[1] = fctInfo[fctNo].up[0];
                fctInfo[fctNo].g_up[1] = fctInfo[fctNo].g_up[0];
                fctInfo[fctNo].localNo[1] = fctInfo[fctNo].localNo[0];
                fctInfo[fctNo].inside[1] = fctInfo[fctNo].inside[0];
            }
            assert(fctInfo[fctNo].inside[0] || fctInfo[fctNo].inside[1]);
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
