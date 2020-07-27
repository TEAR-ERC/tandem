#include "DG.h"
#include "form/RefElement.h"
#include "geometry/Vector.h"
#include "util/Combinatorics.h"
#include "util/Enumerate.h"

#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <utility>

namespace tndm {

template <std::size_t D>
DG<D>::DG(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl,
          std::unique_ptr<RefElement<D>> refElement, unsigned minQuadOrder, MPI_Comm comm)
    : refElement_(std::move(refElement)), numElems_(mesh.numElements()),
      numLocalElems_(mesh.elements().localSize()), numLocalFacets_(mesh.facets().localSize()),
      comm_(comm), elementScatter_(mesh.elements(), comm), fctInfo(mesh.facets().localSize()),
      volInfo(mesh.numElements()) {
    fctRule = simplexQuadratureRule<D - 1u>(minQuadOrder);
    volRule = simplexQuadratureRule<D>(minQuadOrder);

    E = refElement_->evaluateBasisAt(volRule.points());
    D_xi = refElement_->evaluateGradientAt(volRule.points());

    for (std::size_t f = 0; f < D + 1u; ++f) {
        auto points = cl.facetParam(f, fctRule.points());
        e.emplace_back(refElement_->evaluateBasisAt(points));
        d_xi.emplace_back(refElement_->evaluateGradientAt(points));
    }

    fct.setStorage(std::make_shared<fct_t>(numLocalFacets() * fctRule.size()), 0u, numLocalFacets(),
                   fctRule.size());

    vol.setStorage(std::make_shared<vol_t>(numElements() * volRule.size()), 0u, numElements(),
                   volRule.size());

    facetPrecompute(mesh, cl);
    volumePrecompute(mesh, cl);
}

template <std::size_t D>
void DG<D>::facetPrecompute(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl) {
    std::vector<Managed<Matrix<double>>> fctE;
    std::vector<Managed<Tensor<double, 3u>>> fctGradE;

    auto boundaryData = dynamic_cast<BoundaryData const*>(mesh.facets().data());
    if (!boundaryData) {
        throw std::runtime_error("Boundary conditions not set.");
    }

    for (std::size_t f = 0; f < D + 1u; ++f) {
        fctE.emplace_back(cl.evaluateBasisAt(cl.facetParam(f, fctRule.points())));
    }

    for (std::size_t f = 0; f < D + 1u; ++f) {
        fctGradE.emplace_back(cl.evaluateGradientAt(cl.facetParam(f, fctRule.points())));
    }

#pragma omp parallel
    {
        auto J = Managed(cl.jacobianResultInfo(fctRule.size()));
        auto detJ = Managed(cl.detJResultInfo(fctRule.size()));

#pragma omp for
        for (std::size_t fctNo = 0; fctNo < numLocalFacets(); ++fctNo) {
            auto elNos = mesh.template upward<D - 1u>(fctNo);
            assert(elNos.size() >= 1u && elNos.size() <= 2u);
            auto dws = mesh.template downward<D - 1u, D>(elNos[0]);
            auto localFctNo = std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fctNo));
            assert(localFctNo < D + 1u);

            auto jInv = Tensor(fct[fctNo].template get<JInv>().data()->data(),
                               cl.jacobianResultInfo(fctRule.size()));
            auto normal = Tensor(fct[fctNo].template get<Normal>().data()->data(),
                                 cl.normalResultInfo(fctRule.size()));
            auto coords = Tensor(fct[fctNo].template get<Coords>().data()->data(),
                                 cl.mapResultInfo(fctRule.size()));
            cl.jacobian(elNos[0], fctGradE[localFctNo], J);
            cl.detJ(elNos[0], J, detJ);
            cl.jacobianInv(J, jInv);
            cl.normal(localFctNo, detJ, jInv, normal);
            auto& length = fct[fctNo].template get<NormalLength>();
            for (std::size_t i = 0; i < length.size(); ++i) {
                length[i] = norm(fct[fctNo].template get<Normal>()[i]);
            }
            cl.map(elNos[0], fctE[localFctNo], coords);

            double area = 0.0;
            for (std::ptrdiff_t i = 0; i < length.size(); ++i) {
                area += fctRule.weights()[i] * length[i];
            }

            fctInfo[fctNo].up[0] = elNos[0];
            fctInfo[fctNo].g_up[0] = mesh.elements().l2cg(elNos[0]);
            fctInfo[fctNo].localNo[0] = localFctNo;
            fctInfo[fctNo].area = area;
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

                auto jInvOther = Tensor(fct[fctNo].template get<JInvOther>().data()->data(),
                                        cl.jacobianResultInfo(fctRule.size()));

                cl.jacobian(elNos[1], fctGradE[localFNoOther], J);
                cl.jacobianInv(J, jInvOther);
            } else {
                fctInfo[fctNo].up[1] = fctInfo[fctNo].up[0];
                fctInfo[fctNo].g_up[1] = fctInfo[fctNo].g_up[0];
                fctInfo[fctNo].localNo[1] = fctInfo[fctNo].localNo[0];
                fctInfo[fctNo].inside[1] = fctInfo[fctNo].inside[0];
            }
            assert(fctInfo[fctNo].inside[0] || fctInfo[fctNo].inside[1]);
        }
    }
}

template <std::size_t D>
void DG<D>::volumePrecompute(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl) {
    int rank;
    MPI_Comm_rank(comm(), &rank);
    auto geoE = cl.evaluateBasisAt(volRule.points());
    auto gradE = cl.evaluateGradientAt(volRule.points());

#pragma omp parallel
    {
        auto J = Managed(cl.jacobianResultInfo(volRule.size()));

#pragma omp for
        for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
            auto jInv = Tensor(vol[elNo].template get<JInv>().data()->data(),
                               cl.jacobianResultInfo(volRule.size()));
            auto coords = Tensor(vol[elNo].template get<Coords>().data()->data(),
                                 cl.mapResultInfo(volRule.size()));
            auto absDetJ =
                Tensor(vol[elNo].template get<AbsDetJ>().data(), cl.detJResultInfo(volRule.size()));
            cl.jacobian(elNo, gradE, J);
            cl.absDetJ(elNo, J, absDetJ);
            cl.jacobianInv(J, jInv);
            cl.map(elNo, geoE, coords);

            volInfo[elNo].template get<GID>() = mesh.elements().l2cg(elNo);
        }

#pragma omp for
        for (std::size_t elNo = 0; elNo < numLocalElements(); ++elNo) {
            auto absDetJ =
                Tensor(vol[elNo].template get<AbsDetJ>().data(), cl.detJResultInfo(volRule.size()));

            // Compute shape measure for interior penalty method
            // See Shahbazi, "An explicit expression for the penalty parameter of the
            //                interior penalty method"
            double volume = 0.0;
            for (std::ptrdiff_t i = 0; i < volRule.size(); ++i) {
                volume += volRule.weights()[i] * absDetJ(i);
            }
            auto dws = mesh.template downward<D - 1u, D>(elNo);
            double penalty = 0.0;
            for (auto&& fctNo : dws) {
                auto& info = fctInfo[fctNo];
                double half = (info.up[0] != info.up[1]) ? 0.5 : 1.0;
                penalty += info.area * half;
            }
            volInfo[elNo].template get<Penalty>() = penalty / volume;

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

    elementScatter_.scatter(&volInfo[0].template get<Penalty>());
}

template class DG<2u>;
template class DG<3u>;

} // namespace tndm

