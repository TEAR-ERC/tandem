#include "AdapterBase.h"

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

AdapterBase::AdapterBase(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                         RefElement<DomainDimension - 1u> const& space,
                         SimplexQuadratureRule<DomainDimension - 1u> const& quad_rule,
                         std::array<double, DomainDimension> const& up,
                         std::array<double, DomainDimension> const& ref_normal)
    : cl_(std::move(cl)), quad_rule_(quad_rule), up_(up), ref_normal_(ref_normal) {

    e_q = space.evaluateBasisAt(quad_rule_.points());
    e_q_T = space.evaluateBasisAt(quad_rule_.points(), {1, 0});
    minv = space.inverseMassMatrix();

    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto facetParam = cl_->facetParam(f, quad_rule_.points());
        geoDxi_q.emplace_back(cl_->evaluateGradientAt(facetParam));
    }
}

void AdapterBase::begin_preparation(std::size_t numFaultFaces) {
    auto nq = quad_rule_.size();
    fault_.setStorage(std::make_shared<fault_t>(numFaultFaces * nq), 0u, numFaultFaces, nq);
}

void AdapterBase::prepare(std::size_t faultNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch) {
    auto nq = quad_rule_.size();

    auto J = make_scratch_tensor(scratch, cl_->jacobianResultInfo(nq));
    auto JInv = make_scratch_tensor(scratch, cl_->jacobianResultInfo(nq));
    auto detJ = make_scratch_tensor(scratch, cl_->detJResultInfo(nq));
    auto normal = Tensor(fault_[faultNo].template get<UnitNormal>().data()->data(),
                         cl_->normalResultInfo(nq));
    auto fault_basis_q = Tensor(fault_[faultNo].template get<FaultBasis>().data()->data(),
                                cl_->facetBasisResultInfo(nq));
    cl_->jacobian(info.up[0], geoDxi_q[info.localNo[0]], J);
    cl_->detJ(info.up[0], J, detJ);
    cl_->jacobianInv(J, JInv);
    cl_->normal(info.localNo[0], detJ, JInv, normal);
    cl_->normalize(normal);
    for (std::size_t i = 0; i < nq; ++i) {
        auto& sign_flipped = fault_[faultNo].template get<SignFlipped>()[i];
        auto& normal_i = fault_[faultNo].template get<UnitNormal>()[i];
        auto n_ref_dot_n = dot(ref_normal_, normal_i);
        if (std::fabs(n_ref_dot_n) < 10000.0 * std::numeric_limits<double>::epsilon()) {
            throw std::logic_error("Normal and reference normal are almost perpendicular.");
        }
        sign_flipped = n_ref_dot_n < 0;
        if (sign_flipped) {
            normal_i = -1.0 * normal_i;
        }
    }
    cl_->facetBasis(up_, normal, fault_basis_q);
    for (std::size_t q = 0; q < nq; ++q) {
        if (fault_[faultNo].template get<SignFlipped>()[q]) {
            for (std::size_t i = 0; i < fault_basis_q.shape(1); ++i) {
                for (std::size_t j = 0; j < fault_basis_q.shape(0); ++j) {
                    fault_basis_q(i, j, q) *= -1.0;
                }
            }
        }
    }
    scratch.reset();
}

} // namespace tndm
