#include "PoissonAdapter.h"

#include "config.h"
#include "kernels/poisson/tensor.h"
#include "kernels/poisson_adapter/kernel.h"
#include "kernels/poisson_adapter/tensor.h"

#include <cassert>

namespace tndm {

std::size_t PoissonAdapter::traction_block_size() const {
    // Traction needs only one component for Poisson.
    // We still set 2 components here in order to enable a standardised interface,
    // which includes shear and normal components of traction.
    return poisson_adapter::tensor::traction::Shape[0] * 2;
}

void PoissonAdapter::traction(std::size_t faultNo, Matrix<double> const& traction_q,
                              Vector<double>& traction, LinearAllocator<double>&) const {
    std::fill(traction.data(), traction.data() + traction.size(), 0.0);

    poisson_adapter::kernel::evaluate_traction krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.grad_u = traction_q.data();
    krnl.minv = minv.data();
    krnl.traction = traction.data() + poisson_adapter::tensor::traction::Shape[0];
    krnl.n_unit_q = fault_[faultNo].template get<UnitNormal>().data()->data();
    krnl.w = quad_rule_.weights().data();
    krnl.execute();
}

void PoissonAdapter::slip(std::size_t faultNo, Vector<double const>& state,
                          Matrix<double>& slip_q) const {
    assert(slip_q.shape(0) == 1);
    assert(slip_q.shape(1) == poisson_adapter::tensor::slip_q::size());
    poisson_adapter::kernel::evaluate_slip krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.slip = state.data();
    krnl.slip_q = slip_q.data();
    krnl.execute();

    /* We need to flip the sign in the case that the fault normal points opposite
     * to the face's normal.
     */
    for (std::size_t i = 0, nq = quad_rule_.size(); i < nq; ++i) {
        if (fault_[faultNo].template get<SignFlipped>()[i]) {
            slip_q(0, i) = -slip_q(0, i);
        }
    }
}

} // namespace tndm
