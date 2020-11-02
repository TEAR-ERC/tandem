#include "RateAndStateBase.h"

#include "basis/GaussLegendre.h"

#include <algorithm>
#include <memory>

namespace tndm {

RateAndStateBase::RateAndStateBase() : space_(PolynomialDegree - 1, GaussLegendreFactory()) {}

void RateAndStateBase::begin_preparation(std::size_t numFaultFaces,
                                         Curvilinear<DomainDimension> const& cl) {
    auto nbf = space_.numBasisFunctions();
    fault_.setStorage(std::make_shared<fault_t>(numFaultFaces * nbf), 0u, numFaultFaces, nbf);

    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto facetParam = cl.facetParam(f, space_.refNodes());
        geoE_q.emplace_back(cl.evaluateBasisAt(facetParam));
    }
}

void RateAndStateBase::prepare(std::size_t faultNo, Curvilinear<DomainDimension> const& cl,
                               FacetInfo const& info, LinearAllocator&) {
    auto nbf = space_.numBasisFunctions();
    auto coords =
        Tensor(fault_[faultNo].template get<Coords>().data()->data(), cl.mapResultInfo(nbf));
    cl.map(info.up[0], geoE_q[info.localNo[0]], coords);
}

} // namespace tndm
