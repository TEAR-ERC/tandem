#include "AdapterBase.h"

#include "config.h"
#include "form/BC.h"
#include "form/FacetInfo.h"
#include "form/RefElement.h"
#include "geometry/Vector.h"
#include "tensor/EigenMap.h"
#include "tensor/Utility.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <memory>
#include <stdexcept>
#include <utility>

namespace tndm {

AdapterBase::AdapterBase(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                         RefElement<DomainDimension - 1u> const& space,
                         SimplexQuadratureRule<DomainDimension - 1u> const& quad_rule,
                         std::array<double, DomainDimension> const& up,
                         std::array<double, DomainDimension> const& ref_normal)
    : cl_(std::move(cl)), quad_rule_(quad_rule), up_(up), ref_normal_(ref_normal),
      nbf_(space.numBasisFunctions()) {

    e_q = space.evaluateBasisAt(quad_rule_.points());
    e_q_T = space.evaluateBasisAt(quad_rule_.points(), {1, 0});

    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto facetParam = cl_->facetParam(f, quad_rule_.points());
        geoDxi_q.emplace_back(cl_->evaluateGradientAt(facetParam));
    }
}

void AdapterBase::begin_preparation(std::size_t numFaultFaces) {
    auto nq = quad_rule_.size();
    fault_.setStorage(std::make_shared<fault_t>(numFaultFaces * nq), 0u, numFaultFaces, nq);
    auto nbf2 = nbf_ * nbf_;
    mass_.setStorage(std::make_shared<mass_t>(numFaultFaces * nbf2), 0u, numFaultFaces, nbf2);
}

void AdapterBase::prepare(std::size_t faultNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch) {
    auto nq = quad_rule_.size();

    auto J = make_scratch_tensor(scratch, cl_->jacobianResultInfo(nq));
    auto JInv = make_scratch_tensor(scratch, cl_->jacobianResultInfo(nq));
    auto detJ = make_scratch_tensor(scratch, cl_->detJResultInfo(nq));
    auto& fault_data = fault_[faultNo];
    auto& normal_vec = fault_data.template get<Normal>();
    auto normal =
        Tensor(fault_[faultNo].template get<Normal>().data()->data(), cl_->normalResultInfo(nq));
    auto& nl = fault_data.template get<NormalLength>();
    auto fault_basis_q = Tensor(fault_[faultNo].template get<FaultBasis>().data()->data(),
                                cl_->facetBasisResultInfo(nq));
    auto& sign_flipped_vec = fault_data.template get<SignFlipped>();
    auto m = make_scratch_tensor<Matrix<double>>(scratch, nbf_, nbf_);
    auto mInv = Matrix<double>(mass_[faultNo].template get<MInv>().data(), nbf_, nbf_);
    cl_->jacobian(info.up[0], geoDxi_q[info.localNo[0]], J);
    cl_->detJ(info.up[0], J, detJ);
    cl_->jacobianInv(J, JInv);
    cl_->normal(info.localNo[0], detJ, JInv, normal);
    for (std::size_t i = 0; i < nq; ++i) {
        auto& normal_i = normal_vec[i];
        nl[i] = norm(normal_i);

        auto n_ref_dot_n = dot(ref_normal_, normal_i);
        if (std::fabs(n_ref_dot_n) < 10000.0 * std::numeric_limits<double>::epsilon()) {
            throw std::logic_error("Normal and reference normal are almost perpendicular.");
        }
        sign_flipped_vec[i] = n_ref_dot_n < 0;
        if (sign_flipped_vec[i]) {
            normal_i = -1.0 * normal_i;
        }
    }
    cl_->facetBasis(up_, normal, fault_basis_q);
    for (std::size_t q = 0; q < nq; ++q) {
        if (sign_flipped_vec[q]) {
            for (std::size_t i = 0; i < fault_basis_q.shape(1); ++i) {
                for (std::size_t j = 0; j < fault_basis_q.shape(0); ++j) {
                    fault_basis_q(i, j, q) *= -1.0;
                }
            }
        }
    }

    auto const& w = quad_rule_.weights();
    for (std::size_t i = 0; i < nbf_; ++i) {
        for (std::size_t j = 0; j < nbf_; ++j) {
            m(i, j) = 0.0;
            for (std::size_t q = 0; q < nq; ++q) {
                m(i, j) += w[q] * nl[q] * e_q(i, q) * e_q(j, q);
            }
        }
    }
    EigenMap(mInv) = EigenMap(m).inverse();

    scratch.reset();
}

} // namespace tndm
