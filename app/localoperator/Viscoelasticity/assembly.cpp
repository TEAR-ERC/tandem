/*
Viscoelasticity local operator assembly of the bilinear form.

SIPG discontinuous Galerkin matrix assembly: volume, skeleton (interior
facets), Dirichlet boundary, and free-slip boundary contributions.
*/

#include "Viscoelasticity.h"

#include "form/BC.h"
#include "kernels/viscoelasticity/init.h"
#include "kernels/viscoelasticity/kernel.h"
#include "kernels/viscoelasticity/tensor.h"
#include "util/LinearAllocator.h"

namespace tensor = tndm::viscoelasticity::tensor;
namespace init = tndm::viscoelasticity::init;
namespace kernel = tndm::viscoelasticity::kernel;

namespace tndm {

// Assembly of the matrix

bool Viscoelasticity::assemble_volume(std::size_t elNo, Matrix<double>& A00,
                                      LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double Dx_Q[tensor::Dx_Q::size()];

    kernel::Dx_Q dxKrnl;
    dxKrnl.Dx_Q = Dx_Q;
    dxKrnl.Dxi_Q = Dxi_Q.data();
    dxKrnl.G = vol[elNo].get<JInv>().data()->data();
    dxKrnl.execute();

    kernel::assembleVolume krnl;
    krnl.A = A00.data();
    krnl.delta = init::delta::Values;
    krnl.Dx_Q = Dx_Q;
    krnl.A_dt = volPre[elNo].get<A_dt>().data();
    krnl.B_dt = volPre[elNo].get<B_dt>().data();
    krnl.execute();
    return true;
}

bool Viscoelasticity::assemble_skeleton(std::size_t fctNo, FacetInfo const& info,
                                        Matrix<double>& A00, Matrix<double>& A01,
                                        Matrix<double>& A10, Matrix<double>& A11,
                                        LinearAllocator<double>& scratch) const {
    // Physical gradients for both sides
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

    // Traction operators T^x_{kpuq}
    alignas(ALIGNMENT) double trop_q0[tensor::traction_op_q::size(0)];
    alignas(ALIGNMENT) double trop_q1[tensor::traction_op_q::size(1)];
    {
        kernel::assembleTractionOp tOp;
        tOp.delta = init::delta::Values;
        tOp.n_q = fct[fctNo].get<Normal>().data()->data();
        tOp.Dx_q(0) = Dx_q0;
        tOp.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
        tOp.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
        tOp.traction_op_q(0) = trop_q0;
        tOp.execute(0);
        tOp.Dx_q(1) = Dx_q1;
        tOp.A_dt_q(1) = fctPre[fctNo].get<A_dt_q_1>().data();
        tOp.B_dt_q(1) = fctPre[fctNo].get<B_dt_q_1>().data();
        tOp.traction_op_q(1) = trop_q1;
        tOp.execute(1);
    }

    // Lift operators
    alignas(ALIGNMENT) double L_q0[tensor::L_q::size(0)];
    alignas(ALIGNMENT) double L_q1[tensor::L_q::size(1)];
    compute_lift_skeleton(fctNo, info, L_q0, L_q1);

    // Signs follow SIPG convention; epsilon controls symmetry (SIPG=-1, IIPG=0, NIPG=+1).
    kernel::assembleSurface krnl;
    krnl.c00 = -0.5;
    krnl.c01 = -krnl.c00;
    krnl.c10 = 0.5 * epsilon;
    krnl.c11 = -krnl.c10;
    krnl.c20 = penalty(fctNo);
    krnl.c21 = -krnl.c20;
    krnl.w = fctRule.weights().data();
    krnl.traction_op_q(0) = trop_q0;
    krnl.traction_op_q(1) = trop_q1;
    for (unsigned s = 0; s < 2; ++s)
        krnl.E_q(s) = E_q[info.localNo[s]].data();
    krnl.L_q(0) = L_q0;
    krnl.L_q(1) = L_q1;
    krnl.a(0, 0) = A00.data();
    krnl.a(0, 1) = A01.data();
    krnl.a(1, 0) = A10.data();
    krnl.a(1, 1) = A11.data();
    krnl.execute(0, 0);
    krnl.execute(0, 1);
    krnl.execute(1, 0);
    krnl.execute(1, 1);
    return true;
}

bool Viscoelasticity::assemble_boundary(std::size_t fctNo, FacetInfo const& info,
                                        Matrix<double>& A00,
                                        LinearAllocator<double>& scratch) const {
    // Natural, Traction, FreeSlip BCs do not contribute to the matrix
    if (info.bc == BC::Natural || info.bc == BC::Traction || info.bc == BC::FreeSlip) {
        return false;
    }

    alignas(ALIGNMENT) double Dx_q0[tensor::Dx_q::size(0)];
    {
        kernel::Dx_q dxKrnl;
        dxKrnl.Dx_q(0) = Dx_q0;
        dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
        dxKrnl.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
        dxKrnl.execute(0);
    }

    alignas(ALIGNMENT) double trop_q0[tensor::traction_op_q::size(0)];
    {
        kernel::assembleTractionOp tOp;
        tOp.delta = init::delta::Values;
        tOp.n_q = fct[fctNo].get<Normal>().data()->data();
        tOp.Dx_q(0) = Dx_q0;
        tOp.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
        tOp.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
        tOp.traction_op_q(0) = trop_q0;
        tOp.execute(0);
    }

    alignas(ALIGNMENT) double L_q0[tensor::L_q::size(0)];
    compute_lift_boundary(fctNo, info, L_q0);

    kernel::assembleSurface krnl;
    krnl.c00 = -1.0;
    krnl.c10 = epsilon;
    krnl.c20 = penalty(fctNo);
    krnl.w = fctRule.weights().data();
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.L_q(0) = L_q0;
    krnl.traction_op_q(0) = trop_q0;
    krnl.a(0, 0) = A00.data();
    krnl.execute(0, 0);
    return true;
}

bool Viscoelasticity::assemble_boundary_free_slip(std::size_t fctNo, FacetInfo const& info,
                                                  Matrix<double>& A00,
                                                  LinearAllocator<double>& scratch) const {
    if (info.bc != BC::FreeSlip) {
        return false;
    }

    alignas(ALIGNMENT) double Dx_q0[tensor::Dx_q::size(0)];
    {
        kernel::Dx_q dxKrnl;
        dxKrnl.Dx_q(0) = Dx_q0;
        dxKrnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
        dxKrnl.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
        dxKrnl.execute(0);
    }

    alignas(ALIGNMENT) double trop_q0[tensor::traction_op_q::size(0)];
    {
        kernel::assembleTractionOp tOp;
        tOp.delta = init::delta::Values;
        tOp.n_q = fct[fctNo].get<Normal>().data()->data();
        tOp.Dx_q(0) = Dx_q0;
        tOp.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
        tOp.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
        tOp.traction_op_q(0) = trop_q0;
        tOp.execute(0);
    }

    alignas(ALIGNMENT) double L_q0[tensor::L_q::size(0)];
    compute_lift_boundary(fctNo, info, L_q0);

    kernel::assembleSurfaceFreeSlip krnl;
    krnl.Dx_q(0) = Dx_q0;
    krnl.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
    krnl.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.nl_q = fct[fctNo].get<NormalLength>().data();
    krnl.c00 = -1.0;
    krnl.c10 = epsilon;
    krnl.c20 = penalty(fctNo);
    krnl.w = fctRule.weights().data();
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.L_q(0) = L_q0;
    krnl.traction_op_q(0) = trop_q0;
    krnl.a(0, 0) = A00.data();
    krnl.execute(0, 0);
    return true;
}

} // namespace tndm
