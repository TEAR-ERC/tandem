#include "Adapter.h"

#include "config.h"
#include "kernels/viscoelasticity_adapter/init.h"
#include "kernels/viscoelasticity_adapter/kernel.h"
#include "kernels/viscoelasticity_adapter/tensor.h"

#include <cassert>

namespace tndm {

class Viscoelasticity;

template <> std::size_t Adapter<Viscoelasticity>::traction_block_size() const {
    return viscoelasticity_adapter::tensor::traction::Shape[0] *
           viscoelasticity_adapter::tensor::traction::Shape[1];
}

template <>
void Adapter<Viscoelasticity>::traction(std::size_t faultNo, Matrix<double> const& traction_q,
                                        Vector<double>& traction, LinearAllocator<double>&) const {
    assert(traction_q.size() == viscoelasticity_adapter::tensor::traction_q::Size);
    assert(traction.shape(0) == traction_block_size());

    viscoelasticity_adapter::kernel::evaluate_traction krnl;
    krnl.e_q_T = e_q_T.data();
    krnl.fault_basis_q = fault_[faultNo].template get<FaultBasis>().data()->data();
    krnl.traction_q = traction_q.data();
    krnl.minv = mass_[faultNo].template get<MInv>().data();
    krnl.nl_q = fault_[faultNo].template get<NormalLength>().data();
    krnl.traction = traction.data();
    krnl.w = quad_rule_.weights().data();
    krnl.execute();
}

template <>
void Adapter<Viscoelasticity>::slip(std::size_t faultNo, Vector<double const>& state,
                                    Matrix<double>& slip_q) const {
    assert(slip_q.shape(0) == DomainDimension);
    assert(slip_q.shape(1) == viscoelasticity_adapter::tensor::slip_q::Shape[1]);

    viscoelasticity_adapter::kernel::evaluate_slip krnl;
    krnl.copy_slip = viscoelasticity_adapter::init::copy_slip::Values;
    krnl.e_q = e_q.data();
    krnl.fault_basis_q = fault_[faultNo].template get<FaultBasis>().data()->data();
    krnl.slip = state.data();
    krnl.slip_q = slip_q.data();
    krnl.execute();
}

template <>
void Adapter<Viscoelasticity>::moment_rate(std::size_t faultNo, Matrix<double>& moment_rate_vector,
                                           Vector<double const>& slip_rate,
                                           Matrix<double>& mu_field) const {
    assert(moment_rate_vector.shape()[1] == viscoelasticity_adapter::tensor::moment_rate::Shape[0]);
    viscoelasticity_adapter::kernel::evaluate_moment_rate krnl;
    krnl.slip_rate = slip_rate.data();
    krnl.e_q = e_q.data();
    krnl.moment_rate = moment_rate_vector.data();
    krnl.w = quad_rule_.weights().data();
    krnl.nl_q = fault_[faultNo].template get<NormalLength>().data();
    krnl.mu = mu_field.data();
    krnl.execute();
}

} // namespace tndm
