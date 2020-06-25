#include "DG.h"
#include "form/RefElement.h"
#include "geometry/Vector.h"
#include "util/Combinatorics.h"
#include "util/Enumerate.h"

#include <memory>

namespace tndm {

template <std::size_t D>
DG<D>::DG(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl, unsigned degree,
          unsigned minQuadOrder)
    : fctInfo(mesh.numFacets()) {
    fctRule = simplexQuadratureRule<D - 1u>(minQuadOrder);
    volRule = simplexQuadratureRule<D>(minQuadOrder);

    E = dubinerBasisAt(degree, volRule.points());
    D_xi = dubinerBasisGradientAt(degree, volRule.points());

    for (std::size_t f = 0; f < D + 1u; ++f) {
        auto points = cl.facetParam(f, fctRule.points());
        e.emplace_back(dubinerBasisAt(degree, points));
        d_xi.emplace_back(dubinerBasisGradientAt(degree, points));
    }

    fct.setStorage(std::make_shared<fct_t>(mesh.numFacets() * fctRule.size()), 0u, mesh.numFacets(),
                   fctRule.size());

    vol.setStorage(std::make_shared<vol_t>(mesh.numElements() * volRule.size()), 0u,
                   mesh.numElements(), volRule.size());

    facetPrecompute(mesh, cl);
    volumePrecompute(mesh, cl);
}

template <std::size_t D>
void DG<D>::facetPrecompute(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl) {
    std::vector<Managed<Matrix<double>>> fctE;
    std::vector<Managed<Tensor<double, 3u>>> fctGradE;

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
        for (std::size_t fctNo = 0; fctNo < mesh.numFacets(); ++fctNo) {
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
            fctInfo[fctNo].localNo[0] = localFctNo;
            fctInfo[fctNo].area = area;

            if (elNos.size() > 1) {
                auto dwsOther = mesh.template downward<D - 1u, D>(elNos[1]);
                auto localFNoOther = std::distance(
                    dwsOther.begin(), std::find(dwsOther.begin(), dwsOther.end(), fctNo));
                assert(localFNoOther < D + 1u);

                fctInfo[fctNo].up[1] = elNos[1];
                fctInfo[fctNo].localNo[1] = localFNoOther;

                auto jInvOther = Tensor(fct[fctNo].template get<JInvOther>().data()->data(),
                                        cl.jacobianResultInfo(fctRule.size()));

                cl.jacobian(elNos[1], fctGradE[localFNoOther], J);
                cl.jacobianInv(J, jInvOther);
            } else {
                fctInfo[fctNo].up[1] = elNos[0];
                fctInfo[fctNo].localNo[1] = localFctNo;
            }
        }
    }
}

template <std::size_t D>
void DG<D>::volumePrecompute(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl) {
    auto geoE = cl.evaluateBasisAt(volRule.points());
    auto gradE = cl.evaluateGradientAt(volRule.points());

#pragma omp parallel
    {
        auto J = Managed(cl.jacobianResultInfo(volRule.size()));

#pragma omp for
        for (std::size_t elNo = 0; elNo < mesh.numElements(); ++elNo) {
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
        }
    }
}

template class DG<2u>;
template class DG<3u>;

} // namespace tndm

