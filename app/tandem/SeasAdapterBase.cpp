#include "SeasAdapterBase.h"

#include "config.h"
#include "form/BC.h"
#include "form/FacetInfo.h"
#include "form/RefElement.h"
#include "geometry/Vector.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace tndm {

SeasAdapterBase::SeasAdapterBase(
    std::shared_ptr<DGOperatorTopo> topo, std::unique_ptr<RefElement<DomainDimension - 1u>> space,
    std::vector<std::array<double, DomainDimension - 1u>> const& quadPoints,
    std::array<double, DomainDimension> const& ref_normal)
    : topo_(std::move(topo)), space_(std::move(space)), faultMap_(*topo_, BC::Fault),
      ref_normal_(ref_normal) {

    e_q = space_->evaluateBasisAt(quadPoints);
    e_q_T = space_->evaluateBasisAt(quadPoints, {1, 0});
    minv = space_->inverseMassMatrix();
}

void SeasAdapterBase::begin_preparation(std::size_t numFaultFaces,
                                        Curvilinear<DomainDimension> const& cl) {
    auto nbf = space_->numBasisFunctions();
    sign_.setStorage(std::make_shared<sign_t>(numFaultFaces * nbf), 0u, numFaultFaces, nbf);

    auto const space = dynamic_cast<NodalRefElement<DomainDimension - 1u> const*>(space_.get());
    if (space == nullptr) {
        throw std::runtime_error("Nodal basis required for SeasAdapter.");
    }

    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto facetParam = cl.facetParam(f, space->refNodes());
        geoDxi.emplace_back(cl.evaluateGradientAt(facetParam));
    }
}

void SeasAdapterBase::prepare(std::size_t faultNo, Curvilinear<DomainDimension> const& cl,
                              LinearAllocator<double>& scratch) {
    auto const nbf = space_->numBasisFunctions();
    auto const fctNo = faultMap_.fctNo(faultNo);
    auto const& info = topo_->info(fctNo);

    double* J_mem = scratch.allocate(nbf * DomainDimension * DomainDimension);
    double* JInv_mem = scratch.allocate(nbf * DomainDimension * DomainDimension);
    double* detJ_mem = scratch.allocate(nbf);
    auto J = Tensor(J_mem, cl.jacobianResultInfo(nbf));
    auto JInv = Tensor(JInv_mem, cl.jacobianResultInfo(nbf));
    auto detJ = Tensor(detJ_mem, cl.detJResultInfo(nbf));
    auto normal =
        Tensor(sign_[faultNo].template get<UnitNormal>().data()->data(), cl.normalResultInfo(nbf));
    cl.jacobian(info.up[0], geoDxi[info.localNo[0]], J);
    cl.detJ(info.up[0], J, detJ);
    cl.jacobianInv(J, JInv);
    cl.normal(info.localNo[0], detJ, JInv, normal);
    cl.normalize(normal);
    for (std::size_t i = 0; i < nbf; ++i) {
        auto& sign_flipped = sign_[faultNo].template get<SignFlipped>()[i];
        auto& normal_i = sign_[faultNo].template get<UnitNormal>()[i];
        sign_flipped = dot(ref_normal_, normal_i) < 0;
        if (sign_flipped) {
            normal_i = -1.0 * normal_i;
        }
    }
    scratch.reset();
}

} // namespace tndm
