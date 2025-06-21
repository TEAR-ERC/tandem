#include "Adapter.h"

#include "config.h"
#include "kernels/poisson/tensor.h"
#include "kernels/poisson_adapter/kernel.h"
#include "kernels/poisson_adapter/tensor.h"

#include <cassert>

namespace tndm {

class Poisson;

template <> std::size_t Adapter<Poisson>::traction_block_size() const {
    // Traction needs only one component for Poisson.
    // We still set 2 components here in order to enable a standardised interface,
    // which includes shear and normal components of traction.
    return poisson_adapter::tensor::traction::Shape[0] * 2;
}

template <>
void Adapter<Poisson>::traction(std::size_t faultNo, Matrix<double> const& traction_q,
                                Vector<double>& traction, LinearAllocator<double>&) const {
    std::fill(traction.data(), traction.data() + traction.size(), 0.0);

    poisson_adapter::kernel::evaluate_traction krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.grad_u = traction_q.data();
    krnl.minv = mass_[faultNo].template get<MInv>().data();
    krnl.traction = traction.data() + poisson_adapter::tensor::traction::Shape[0];
    krnl.n_q = fault_[faultNo].template get<Normal>().data()->data();
    krnl.w = quad_rule_.weights().data();
    krnl.execute();
}

template <>
void Adapter<Poisson>::slip(std::size_t faultNo, Vector<double const>& state,
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

template <>
void Adapter<Poisson>::slip_rate(std::size_t faultNo, Vector<double const>& state,
                                 Matrix<double>& slip_rate_q) const {
    assert(slip_rate_q.shape(0) == DomainDimension);
    assert(slip_rate_q.shape(1) == poisson_adapter::tensor::slip_rate_q::Shape[1]);
    poisson_adapter::kernel::evaluate_slip_rate krnl;
    krnl.slip_rate = state.data();
    krnl.e_q = e_q.data();
    krnl.slip_rate_q = slip_rate_q.data();
    krnl.execute();
}

template <>
void Adapter<Poisson>::moment_rate(std::size_t faultNo, Matrix<double>& moment_rate_vector,
                                   Matrix<double>& slip_rate_q, double* mu_field) const {
    // TODO: Add lame parameter mu during moment rate calculation
    assert(moment_rate_vector.shape()[1] == poisson_adapter::tensor::moment_rate::Shape[0]);
    poisson_adapter::kernel::evaluate_moment_rate krnl;
    krnl.slip_rate_q = slip_rate_q.data();
    krnl.moment_rate = moment_rate_vector.data();
    krnl.w = quad_rule_.weights().data();
    krnl.nl_q = fault_[faultNo].template get<NormalLength>().data();
    krnl.execute();
}

} // namespace tndm
