#include "SeasAdapterBase.h"

#include "config.h"
#include "form/BC.h"
#include "form/FacetInfo.h"
#include "form/RefElement.h"
#include "geometry/Vector.h"
#include "tensor/Utility.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace tndm {

SeasAdapterBase::SeasAdapterBase(
    std::shared_ptr<Curvilinear<DomainDimension>> cl, std::shared_ptr<DGOperatorTopo> topo,
    std::unique_ptr<RefElement<DomainDimension - 1u>> space,
    std::vector<std::array<double, DomainDimension - 1u>> const& quadPoints,
    std::array<double, DomainDimension> const& up,
    std::array<double, DomainDimension> const& ref_normal)
    : cl_(std::move(cl)), topo_(std::move(topo)), space_(std::move(space)),
      faultMap_(*topo_, BC::Fault), up_(up), ref_normal_(ref_normal), nq_(quadPoints.size()) {

    e_q = space_->evaluateBasisAt(quadPoints);
    e_q_T = space_->evaluateBasisAt(quadPoints, {1, 0});
    minv = space_->inverseMassMatrix();

    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto facetParam = cl_->facetParam(f, quadPoints);
        geoDxi_q.emplace_back(cl_->evaluateGradientAt(facetParam));
    }
}

void SeasAdapterBase::begin_preparation(std::size_t numFaultFaces) {
    fault_.setStorage(std::make_shared<fault_t>(numFaultFaces * nq_), 0u, numFaultFaces, nq_);
}

void SeasAdapterBase::prepare(std::size_t faultNo, LinearAllocator<double>& scratch) {
    auto const fctNo = faultMap_.fctNo(faultNo);
    auto const& info = topo_->info(fctNo);

    auto J = make_scratch_tensor(scratch, cl_->jacobianResultInfo(nq_));
    auto JInv = make_scratch_tensor(scratch, cl_->jacobianResultInfo(nq_));
    auto detJ = make_scratch_tensor(scratch, cl_->detJResultInfo(nq_));
    auto normal = Tensor(fault_[faultNo].template get<UnitNormal>().data()->data(),
                         cl_->normalResultInfo(nq_));
    auto fault_basis_q = Tensor(fault_[faultNo].template get<FaultBasis>().data()->data(),
                                cl_->facetBasisResultInfo(nq_));
    cl_->jacobian(info.up[0], geoDxi_q[info.localNo[0]], J);
    cl_->detJ(info.up[0], J, detJ);
    cl_->jacobianInv(J, JInv);
    cl_->normal(info.localNo[0], detJ, JInv, normal);
    cl_->normalize(normal);
    for (std::size_t i = 0; i < nq_; ++i) {
        auto& sign_flipped = fault_[faultNo].template get<SignFlipped>()[i];
        auto& normal_i = fault_[faultNo].template get<UnitNormal>()[i];
        auto up_dot_n = dot(up_, normal_i);
        if (std::fabs(up_dot_n) > 10.0 * std::numeric_limits<double>::epsilon()) {
            sign_flipped = up_dot_n < 0;
        } else {
            sign_flipped = dot(ref_normal_, normal_i) < 0;
        }
        if (sign_flipped) {
            normal_i = -1.0 * normal_i;
        }
    }
    cl_->facetBasis(up_, normal, fault_basis_q);
    scratch.reset();
}

} // namespace tndm
