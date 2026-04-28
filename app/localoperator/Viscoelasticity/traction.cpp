/*
Viscoelasticity_traction.cpp

Traction computation at fault interfaces and boundaries, used by the SEAS
friction law adapter.
*/

#include "Viscoelasticity.h"

#include "kernels/viscoelasticity/init.h"
#include "kernels/viscoelasticity/kernel.h"
#include "kernels/viscoelasticity/tensor.h"

namespace tensor = tndm::viscoelasticity::tensor;
namespace init = tndm::viscoelasticity::init;
namespace kernel = tndm::viscoelasticity::kernel;

namespace tndm {

// Traction computation — fault interface (adapter)
TensorBase<Matrix<double>> Viscoelasticity::tractionResultInfo() const {
    return TensorBase<Matrix<double>>(tensor::traction_q::Shape[0], tensor::traction_q::Shape[1]);
}

void Viscoelasticity::traction_skeleton(std::size_t fctNo, FacetInfo const& info,
                                        Vector<double const>& u0, Vector<double const>& u1,
                                        Matrix<double>& result) const {
    alignas(ALIGNMENT) double Dx_q0[tensor::Dx_q::size(0)];
    alignas(ALIGNMENT) double Dx_q1[tensor::Dx_q::size(1)];
    {
        kernel::Dx_q dxKrnl;
        dxKrnl.Dx_q(0) = Dx_q0;
        dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
        dxKrnl.Dx_q(1) = Dx_q1;
        dxKrnl.g(1) = fct[fctNo].get<JInv1>().data()->data();
        for (unsigned s = 0; s < 2; ++s) {
            dxKrnl.Dxi_q(s) = Dxi_q[info.localNo[s]].data();
            dxKrnl.execute(s);
        }
    }

    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    bc_skeleton(fctNo, info.bc, f_q_raw);

    kernel::compute_traction krnl;
    krnl.c00 = -penalty(fctNo);
    krnl.Dx_q(0) = Dx_q0;
    krnl.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
    krnl.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
    krnl.Dx_q(1) = Dx_q1;
    krnl.A_dt_q(1) = fctPre[fctNo].get<A_dt_q_1>().data();
    krnl.B_dt_q(1) = fctPre[fctNo].get<B_dt_q_1>().data();
    krnl.g_dt_q = fctPre[fctNo].get<g_dt_q>().data();
    krnl.ratio_q = fctPre[fctNo].get<ratio_q>().data();
    krnl.mu1_q(0) = fctPre[fctNo].get<mu1_q_0>().data();
    krnl.mu1_q(1) = fctPre[fctNo].get<mu1_q_1>().data();
    krnl.partial_strain_tensor_q(0) =
        strainHistory_q[fctNo].get<partial_strain_old_q_0>().data()->data();
    krnl.partial_strain_tensor_q(1) =
        strainHistory_q[fctNo].get<partial_strain_old_q_1>().data()->data();
    krnl.deviatoric_strain_tensor_q(0) =
        strainHistory_q[fctNo].get<deviatoric_strain_old_q_0>().data()->data();
    krnl.deviatoric_strain_tensor_q(1) =
        strainHistory_q[fctNo].get<deviatoric_strain_old_q_1>().data()->data();
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.E_q(1) = E_q[info.localNo[1]].data();
    krnl.f_q = f_q_raw;
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.traction_q = result.data();
    krnl.u(0) = u0.data();
    krnl.u(1) = u1.data();
    krnl.execute();
}

void Viscoelasticity::traction_boundary(std::size_t fctNo, FacetInfo const& info,
                                        Vector<double const>& u0, Matrix<double>& result) const {
    alignas(ALIGNMENT) double Dx_q0[tensor::Dx_q::size(0)];
    {
        kernel::Dx_q dxKrnl;
        dxKrnl.Dx_q(0) = Dx_q0;
        dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
        dxKrnl.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
        dxKrnl.execute(0);
    }

    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    bc_boundary(fctNo, info.bc, f_q_raw);

    kernel::compute_traction_bnd krnl;
    krnl.c00 = -penalty(fctNo);
    krnl.Dx_q(0) = Dx_q0;
    krnl.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
    krnl.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
    krnl.g_dt_q = fctPre[fctNo].get<g_dt_q>().data();
    krnl.ratio_q = fctPre[fctNo].get<ratio_q>().data();
    krnl.mu1_q(0) = fctPre[fctNo].get<mu1_q_0>().data();
    krnl.partial_strain_tensor_q(0) =
        strainHistory_q[fctNo].get<partial_strain_old_q_0>().data()->data();
    krnl.deviatoric_strain_tensor_q(0) =
        strainHistory_q[fctNo].get<deviatoric_strain_old_q_0>().data()->data();
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.f_q = f_q_raw;
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.traction_q = result.data();
    krnl.u(0) = u0.data();

    krnl.execute();
}

} // namespace tndm
