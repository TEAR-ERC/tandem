#include "SeasPoissonAdapter.h"

#include "kernels/poisson/tensor.h"
#include "kernels/poisson_adapter/kernel.h"
#include "kernels/poisson_adapter/tensor.h"

#include "form/FacetInfo.h"
#include "form/RefElement.h"
#include "tandem/SeasAdapterBase.h"
#include "tensor/Managed.h"
#include "tensor/Utility.h"

#include <cassert>

namespace tndm {
SeasPoissonAdapter::SeasPoissonAdapter(std::shared_ptr<Curvilinear<Dim>> cl,
                                       std::shared_ptr<DGOperatorTopo> topo,
                                       std::unique_ptr<RefElement<Dim - 1u>> space,
                                       std::unique_ptr<Poisson> local_operator,
                                       std::array<double, Dim> const& ref_normal,
                                       double normal_stress)
    : SeasAdapterBase(std::move(cl), topo, std::move(space),
                      local_operator->facetQuadratureRule().points(), ref_normal),
      dgop_(std::make_unique<DGOperator<Poisson>>(std::move(topo), std::move(local_operator))),
      linear_solver_(*dgop_), normal_stress_(normal_stress) {}

void SeasPoissonAdapter::slip(std::size_t faultNo, Vector<double const>& state,
                              Matrix<double>& slip_q) const {
    assert(slip_q.shape(0) == 1);
    assert(slip_q.shape(1) == poisson_adapter::tensor::slip_q::size());
    poisson_adapter::kernel::evaluate_slip krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.slip = state.data();
    krnl.slip_q = slip_q.data();
    krnl.execute();

    for (std::size_t i = 0; i < nq_; ++i) {
        if (!fault_[faultNo].template get<SignFlipped>()[i]) {
            slip_q(0, i) = -slip_q(0, i);
        }
    }
}

TensorBase<Matrix<double>> SeasPoissonAdapter::traction_info() const {
    return TensorBase<Matrix<double>>(poisson_adapter::tensor::traction::Shape[0], 2);
}

void SeasPoissonAdapter::traction(std::size_t faultNo, Matrix<double>& traction,
                                  LinearAllocator<double>&) const {
    double grad_u_raw[poisson::tensor::grad_u::Size];
    auto grad_u = Matrix<double>(grad_u_raw, dgop_->lop().tractionResultInfo());
    assert(grad_u.size() == poisson::tensor::grad_u::Size);

    auto fctNo = faultMap_.fctNo(faultNo);
    auto const& info = dgop_->topo().info(fctNo);
    auto u0 = linear_solver_.x().get_block(handle_, info.up[0]);
    auto u1 = linear_solver_.x().get_block(handle_, info.up[1]);
    dgop_->lop().traction(fctNo, info, u0, u1, grad_u);
    poisson_adapter::kernel::evaluate_traction krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.grad_u = grad_u_raw;
    krnl.minv = minv.data();
    krnl.traction = &traction(0, 1);
    krnl.n_unit_q = fault_[faultNo].template get<UnitNormal>().data()->data();
    krnl.w = dgop_->lop().facetQuadratureRule().weights().data();
    krnl.execute();

    auto const nbf = space_->numBasisFunctions();
    for (std::size_t i = 0; i < nbf; ++i) {
        traction(i, 0) = normal_stress_;
    }
}

} // namespace tndm
