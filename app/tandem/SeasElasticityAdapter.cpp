#include "SeasElasticityAdapter.h"

#include "config.h"
#include "geometry/Curvilinear.h"
#include "kernels/elasticity/tensor.h"
#include "kernels/elasticity_adapter/init.h"
#include "kernels/elasticity_adapter/kernel.h"
#include "kernels/elasticity_adapter/tensor.h"

#include "form/FacetInfo.h"
#include "form/RefElement.h"
#include "localoperator/Elasticity.h"
#include "tensor/Managed.h"

#include <cassert>

namespace tndm {
SeasElasticityAdapter::SeasElasticityAdapter(std::shared_ptr<Curvilinear<Dim>> cl,
                                             std::shared_ptr<DGOperatorTopo> topo,
                                             std::unique_ptr<RefElement<Dim - 1u>> space,
                                             std::unique_ptr<Elasticity> local_operator,
                                             std::array<double, Dim> const& up,
                                             std::array<double, Dim> const& ref_normal)
    : SeasAdapterBase(std::move(cl), topo, std::move(space),
                      local_operator->facetQuadratureRule().points(), up, ref_normal),
      dgop_(std::make_unique<DGOperator<Elasticity>>(std::move(topo), std::move(local_operator))),
      linear_solver_(*dgop_) {}

void SeasElasticityAdapter::slip(std::size_t faultNo, Vector<double const>& state,
                                 Matrix<double>& slip_q) const {
    assert(slip_q.shape(0) == DomainDimension);
    assert(slip_q.shape(1) == elasticity_adapter::tensor::slip_q::Shape[1]);

    elasticity_adapter::kernel::evaluate_slip krnl;
    krnl.copy_slip = elasticity_adapter::init::copy_slip::Values;
    krnl.e_q = e_q.data();
    krnl.fault_basis_q = fault_[faultNo].template get<FaultBasis>().data()->data();
    krnl.slip = state.data();
    krnl.slip_q = slip_q.data();
    krnl.execute();

    for (std::size_t i = 0; i < nq_; ++i) {
        /* Slip in the Elasticity solver is defined as [[u]] := u^- - u^+.
         * In the friction solver the sign of slip S is flipped, that is, S = -[[u]].
         */
        if (!fault_[faultNo].template get<SignFlipped>()[i]) {
            for (std::size_t d = 0; d < DomainDimension; ++d) {
                slip_q(d, i) = -slip_q(d, i);
            }
        }
    }
}

TensorBase<Matrix<double>> SeasElasticityAdapter::traction_info() const {
    return TensorBase<Matrix<double>>(elasticity_adapter::tensor::traction::Shape[0],
                                      elasticity_adapter::tensor::traction::Shape[1]);
}

void SeasElasticityAdapter::traction(std::size_t faultNo, Matrix<double>& traction,
                                     LinearAllocator<double>&) const {
    auto const nbf = space_->numBasisFunctions();
    assert(traction.shape(0) == nbf);
    assert(traction.shape(1) == DomainDimension);

    double traction_q_raw[elasticity::tensor::traction_q::Size];
    auto traction_q = Matrix<double>(traction_q_raw, dgop_->lop().tractionResultInfo());
    assert(traction_q.size() == elasticity::tensor::traction_q::Size);

    auto fctNo = faultMap_.fctNo(faultNo);
    auto const& info = dgop_->topo().info(fctNo);
    auto u0 = linear_solver_.x().get_block(handle_, info.up[0]);
    auto u1 = linear_solver_.x().get_block(handle_, info.up[1]);
    if (info.up[0] == info.up[1]) {
        dgop_->lop().traction_boundary(fctNo, info, u0, traction_q);
    } else {
        dgop_->lop().traction_skeleton(fctNo, info, u0, u1, traction_q);
    }
    elasticity_adapter::kernel::evaluate_traction krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.fault_basis_q = fault_[faultNo].template get<FaultBasis>().data()->data();
    krnl.traction_q = traction_q_raw;
    krnl.minv = minv.data();
    krnl.traction = traction.data();
    krnl.w = dgop_->lop().facetQuadratureRule().weights().data();
    krnl.execute();
}

} // namespace tndm
