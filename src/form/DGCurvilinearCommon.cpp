#include "DGCurvilinearCommon.h"

#include "geometry/Vector.h"
#include "parallel/SimpleScatter.h"
#include "quadrules/AutoRule.h"

#include <memory>
#include <utility>

namespace tndm {

template <std::size_t D>
DGCurvilinearCommon<D>::DGCurvilinearCommon(std::shared_ptr<Curvilinear<D>> cl,
                                            unsigned minQuadOrder)
    : cl_(std::move(cl)) {
    fctRule = simplexQuadratureRule<D - 1u>(minQuadOrder);
    volRule = simplexQuadratureRule<D>(minQuadOrder);

    geoE_Q = cl_->evaluateBasisAt(volRule.points());
    geoDxi_Q = cl_->evaluateGradientAt(volRule.points());
    for (std::size_t f = 0; f < D + 1u; ++f) {
        auto facetParam = cl_->facetParam(f, fctRule.points());
        geoE_q.emplace_back(cl_->evaluateBasisAt(facetParam));
        geoDxi_q.emplace_back(cl_->evaluateGradientAt(facetParam));
    }
}

template <std::size_t D>
void DGCurvilinearCommon<D>::begin_preparation(std::size_t numElements,
                                               std::size_t numLocalElements,
                                               std::size_t numLocalFacets) {
    fct.setStorage(std::make_shared<fct_t>(numLocalFacets * fctRule.size()), 0u, numLocalFacets,
                   fctRule.size());
    vol.setStorage(std::make_shared<vol_t>(numElements * volRule.size()), 0u, numElements,
                   volRule.size());
    area_.resize(numLocalFacets);
    volume_.resize(numElements);
}

template <std::size_t D>
void DGCurvilinearCommon<D>::prepare_volume(std::size_t elNo, LinearAllocator<double>& scratch) {
    double* Jmem = scratch.allocate(volRule.size() * D * D);
    auto J = Tensor(Jmem, cl_->jacobianResultInfo(volRule.size()));
    auto jInv = Tensor(vol[elNo].template get<JInv>().data()->data(),
                       cl_->jacobianResultInfo(volRule.size()));
    auto coords =
        Tensor(vol[elNo].template get<Coords>().data()->data(), cl_->mapResultInfo(volRule.size()));
    auto absDetJ =
        Tensor(vol[elNo].template get<AbsDetJ>().data(), cl_->detJResultInfo(volRule.size()));
    auto volTag = Tensor(vol[elNo].template get<volumeTag>().data(), cl_->tagsInfo(volRule.size()));

    cl_->jacobian(elNo, geoDxi_Q, J);
    cl_->absDetJ(elNo, J, absDetJ);
    cl_->jacobianInv(J, jInv);
    cl_->map(elNo, geoE_Q, coords);
    cl_->setVolumeTags(elNo, volTag);

    double volume = 0.0;
    for (std::ptrdiff_t i = 0; i < volRule.size(); ++i) {
        volume += volRule.weights()[i] * absDetJ(i);
    }
    volume_[elNo] = volume;
}

template <std::size_t D>
void DGCurvilinearCommon<D>::prepare_bndskl(std::size_t fctNo, FacetInfo const& info,
                                            LinearAllocator<double>& scratch) {
    double* Jmem = scratch.allocate(fctRule.size() * D * D);
    double* detJmem = scratch.allocate(fctRule.size());
    auto J = Tensor(Jmem, cl_->jacobianResultInfo(fctRule.size()));
    auto detJ = Tensor(detJmem, cl_->detJResultInfo(fctRule.size()));

    auto jInv0 = Tensor(fct[fctNo].template get<JInv0>().data()->data(),
                        cl_->jacobianResultInfo(fctRule.size()));
    auto normal = Tensor(fct[fctNo].template get<Normal>().data()->data(),
                         cl_->normalResultInfo(fctRule.size()));
    auto unit_normal = Tensor(fct[fctNo].template get<UnitNormal>().data()->data(),
                              cl_->normalResultInfo(fctRule.size()));
    auto coords = Tensor(fct[fctNo].template get<Coords>().data()->data(),
                         cl_->mapResultInfo(fctRule.size()));
    auto facetTags =
        Tensor(fct[fctNo].template get<facetTag>().data(), cl_->tagsInfo(fctRule.size()));

    cl_->jacobian(info.up[0], geoDxi_q[info.localNo[0]], J);
    cl_->detJ(info.up[0], J, detJ);
    cl_->jacobianInv(J, jInv0);
    cl_->normal(info.localNo[0], detJ, jInv0, normal);
    auto& length = fct[fctNo].template get<NormalLength>();
    for (std::size_t i = 0; i < length.size(); ++i) {
        length[i] = norm(fct[fctNo].template get<Normal>()[i]);
    }
    cl_->normal(info.localNo[0], detJ, jInv0, unit_normal);
    cl_->normalize(unit_normal);
    cl_->map(info.up[0], geoE_q[info.localNo[0]], coords);
    cl_->setFacetTags(info, facetTags);

    double area = 0.0;
    for (std::ptrdiff_t i = 0; i < length.size(); ++i) {
        area += fctRule.weights()[i] * length[i];
    }
    area_[fctNo] = area;

    auto jInv1 = Tensor(fct[fctNo].template get<JInv1>().data()->data(),
                        cl_->jacobianResultInfo(fctRule.size()));

    cl_->jacobian(info.up[1], geoDxi_q[info.localNo[1]], J);
    cl_->jacobianInv(J, jInv1);
}

template <std::size_t D>
void DGCurvilinearCommon<D>::end_preparation(std::shared_ptr<ScatterPlan> elementScatterPlan) {
    auto scatter = SimpleScatter<double>(std::move(elementScatterPlan));
    scatter.scatter(volume_.data());
}

template class DGCurvilinearCommon<2u>;
template class DGCurvilinearCommon<3u>;

} // namespace tndm
