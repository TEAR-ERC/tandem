#include "SeasPoissonAdapter.h"

#include "kernels/poisson/tensor.h"
#include "kernels/rate_and_state/kernel.h"
#include "kernels/rate_and_state/tensor.h"

#include "form/FacetInfo.h"
#include "form/RefElement.h"
#include "tensor/Managed.h"

#include <cassert>

namespace tndm {

void SeasPoissonAdapter::slip(std::size_t faultNo, Vector<double const>& state,
                              Matrix<double>& s_q) const {
    auto const nbf = space_.numBasisFunctions();
    double slip_flip[rate_and_state::tensor::slip::Size];
    double const* slip = state.data() + nbf;
    assert(rate_and_state::tensor::slip::Size == nbf);

    for (std::size_t i = 0; i < nbf; ++i) {
        if (!fault_[faultNo].template get<SignFlipped>()[i]) {
            slip_flip[i] = -slip[i];
        } else {
            slip_flip[i] = slip[i];
        }
    }
    assert(s_q.shape(0) == 1);
    assert(s_q.shape(1) == rate_and_state::tensor::slip_q::size());
    rate_and_state::kernel::evaluate_slip krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.slip = slip_flip;
    krnl.slip_q = s_q.data();
    krnl.execute();
}

TensorBase<Matrix<double>> SeasPoissonAdapter::traction_info() const {
    return TensorBase<Matrix<double>>(rate_and_state::tensor::traction::Shape[0], 1);
}

void SeasPoissonAdapter::traction(std::size_t faultNo, Matrix<double>& traction,
                                  LinearAllocator&) const {
    double grad_u_raw[poisson::tensor::grad_u::Size];
    auto grad_u = Matrix<double>(grad_u_raw, dgop_->lop().tractionResultInfo());
    assert(grad_u.size() == poisson::tensor::grad_u::Size);

    auto fctNo = faultMap_.fctNo(faultNo);
    auto const& info = dgop_->topo().info(fctNo);
    auto u0 = linear_solver_.x().get_block(handle_, info.up[0]);
    auto u1 = linear_solver_.x().get_block(handle_, info.up[1]);
    dgop_->lop().traction(fctNo, info, u0, u1, grad_u);
    rate_and_state::kernel::evaluate_traction krnl;
    krnl.grad_u = grad_u_raw;
    krnl.traction = traction.data();
    krnl.unit_normal = fault_[faultNo].template get<UnitNormal>().data()->data();
    krnl.execute();
}

} // namespace tndm
