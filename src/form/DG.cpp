#include "DG.h"
#include "form/RefElement.h"
#include "util/Combinatorics.h"

namespace tndm {

template <std::size_t D>
DG<D>::DG(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl, unsigned degree,
          unsigned minQuadOrder)
    : fctInfo(mesh.numFacets()) {
    fctRule = simplexQuadratureRule<D - 1u>(minQuadOrder);
    volRule = simplexQuadratureRule<D>(minQuadOrder);

    Dxi = dubinerBasisGradientAt(degree, volRule.points());

    E = cl.evaluateBasisAt(volRule.points());
    gradE = cl.evaluateGradientAt(volRule.points());

    for (std::size_t f = 0; f < D + 1u; ++f) {
        fctE.emplace_back(cl.evaluateBasisAt(cl.facetParam(f, fctRule.points())));
    }

    for (std::size_t f = 0; f < D + 1u; ++f) {
        fctGradE.emplace_back(cl.evaluateGradientAt(cl.facetParam(f, fctRule.points())));
    }

    fctStore.resize(mesh.numFacets() * fctRule.size());
    fct.setStorage(fctStore, 0u, mesh.numFacets(), fctRule.size());

    volStore.resize(mesh.numElements() * volRule.size());
    vol.setStorage(volStore, 0u, mesh.numElements(), volRule.size());

    facetPrecompute(mesh, cl);
    volumePrecompute(mesh, cl);
}

template <std::size_t D>
void DG<D>::facetPrecompute(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl) {
#pragma omp parallel
    {
        auto J = Managed(cl.jacobianResultInfo(fctRule.size()));
        auto detJ = Managed(cl.detJResultInfo(fctRule.size()));

#pragma omp for
        for (std::size_t fctNo = 0; fctNo < mesh.numFacets(); ++fctNo) {
            auto elNos = mesh.template upward<D - 1u>(fctNo);
            assert(elNos.size() >= 1u && elNos.size() <= 2u);
            auto dws = mesh.template downward<D - 1u, D>(elNos[0]);
            auto localFctNo = std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fctNo));

            auto jInvT = Tensor(fct[fctNo].template get<JInvT>().data()->data(),
                                cl.jacobianResultInfo(fctRule.size()));
            auto normal = Tensor(fct[fctNo].template get<Normal>().data()->data(),
                                 cl.normalResultInfo(fctRule.size()));
            cl.jacobian(elNos[0], fctGradE[localFctNo], J);
            cl.detJ(elNos[0], J, detJ);
            cl.jacobianInvT(J, jInvT);
            cl.normal(localFctNo, detJ, jInvT, normal);

            fctInfo[fctNo].up[0] = elNos[0];
            fctInfo[fctNo].localNo[0] = localFctNo;

            if (elNos.size() > 1) {
                auto dwsOther = mesh.template downward<D - 1u, D>(elNos[1]);
                auto localFNoOther = std::distance(
                    dwsOther.begin(), std::find(dwsOther.begin(), dwsOther.end(), fctNo));

                fctInfo[fctNo].up[1] = elNos[1];
                fctInfo[fctNo].localNo[1] = localFNoOther;
            } else {
                fctInfo[fctNo].up[1] = elNos[0];
                fctInfo[fctNo].localNo[1] = localFctNo;
            }
        }
    }
}

template <std::size_t D>
void DG<D>::volumePrecompute(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl) {
#pragma omp parallel
    {
        auto J = Managed(cl.jacobianResultInfo(volRule.size()));

#pragma omp for
        for (std::size_t elNo = 0; elNo < mesh.numElements(); ++elNo) {
            auto jInvT = Tensor(vol[elNo].template get<JInvT>().data()->data(),
                                cl.jacobianResultInfo(volRule.size()));
            auto absDetJ =
                Tensor(vol[elNo].template get<AbsDetJ>().data(), cl.detJResultInfo(volRule.size()));
            cl.jacobian(elNo, gradE, J);
            cl.absDetJ(elNo, J, absDetJ);
            cl.jacobianInvT(J, jInvT);
        }
    }
}

template class DG<2u>;
template class DG<3u>;

} // namespace tndm

