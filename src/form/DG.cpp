#include "DG.h"

namespace tndm {

template <std::size_t D>
DG<D>::DG(LocalSimplexMesh<D> const& mesh, Curvilinear<D> const& cl, unsigned minQuadOrder)
    : cl(cl), fctInfo(mesh->numFacets()) {
    fctRule = simplexQuadratureRule<D - 1u>(minQuadOrder);
    volRule = simplexQuadratureRule<D>(minQuadOrder);

    E = cl.evaluateBasisAt(volRule.points());
    gradE = cl.evaluateGradientAt(volRule.points());

    for (std::size_t f = 0; f < D + 1u; ++f) {
        fctE.emplace_back(cl.evaluateBasisAt(cl.facetParam(f, fctRule.points())));
    }

    for (std::size_t f = 0; f < D + 1u; ++f) {
        fctGradE.emplace_back(cl.evaluateGradientAt(cl.facetParam(f, fctRule.points())));
    }

    fctStore.resize(mesh.numFacets() * fctRule.size());
    fct.setStorage(volStore, 0, mesh.numFacets(), fctRule.size());

    volStore.resize(mesh.numElements() * volRule.size());
    vol.setStorage(volStore, 0, mesh.numElements(), volRule.size());

    facetPrecompute(mesh);
    volumePrecompute(mesh);
}

template <std::size_t D> void DG<D>::facetPrecompute(LocalSimplexMesh<D> const& mesh) {
#pragma omp parallel
    {
        auto J = Managed(cl.jacobianResultInfo(fctRule.size()));

#pragma omp for
        for (std::size_t fctNo = 0; fctNo < mesh.numFacets(); ++fctNo) {
            auto elNos = mesh.template upward<D - 1u>(fctNo);
            assert(elNos.size() >= 1u && elNos.size() <= 2u);
            auto dws = mesh->template downward<D - 1u, D>(elNos[0]);
            auto localFctNo = std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fctNo));

            auto tJinvT = Tensor(fct[fctNo].template get<JinvT>().data(),
                                 cl.jacobianResultInfo(fctRule.size()));
            auto tDetJ =
                Tensor(fct[fctNo].template get<detJ>().data(), cl.detJResultInfo(fctRule.size()));
            auto tNormal = Tensor(fct[fctNo].template get<normal>().data(),
                                  cl.normalResultInfo(fctRule.size()));
            cl.detJ(elNos[0], J, tDetJ);
            cl.jacobianInvT(J, tJinvT);
            cl.normal(localFctNo, tDetJ, tJinvT, tNormal);

            fctInfo[fctNo].up[0] = elNos[0];
            fctInfo[fctNo].localNo[0] = localFctNo;

            if (elNos.size() > 1) {
                auto dwsOther = mesh->template downward<D - 1u, D>(elNos[1]);
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

template <std::size_t D> void DG<D>::volumePrecompute(LocalSimplexMesh<D> const& mesh) {
#pragma omp parallel
    {
        auto J = Managed(cl.jacobianResultInfo(volRule.size()));

#pragma omp for
        for (std::size_t elNo = 0; elNo < mesh.numElements(); ++elNo) {
            auto tJinvT = Tensor(vol[elNo].template get<JinvT>().data(),
                                 cl.jacobianResultInfo(volRule.size()));
            auto tDetJ =
                Tensor(vol[elNo].template get<detJ>().data(), cl.detJResultInfo(volRule.size()));
            auto tNormal = Tensor(vol[elNo].template get<normal>().data(),
                                  cl.normalResultInfo(volRule.size()));
            cl.detJ(elNo, J, tDetJ);
            cl.jacobianInvT(J, tJinvT);
        }
    }
}

} // namespace tndm

