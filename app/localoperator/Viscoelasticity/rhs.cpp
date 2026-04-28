/*
rhs assembly for the linear form on the RHS of the viscoelasticity problem.

Right-hand side assembly: body force, Dirichlet/fault/traction/free-slip BCs,
history surface terms, and the shared lift helper for RHS.
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

// BC evaluation helpers — fill f_q_raw from prescribed BC functionals
bool Viscoelasticity::bc_skeleton(std::size_t fctNo, BC bc, double f_q_raw[]) const {
    auto f_q = Matrix<double>(f_q_raw, NumQuantities, fctRule.size());
    if (bc == BC::Fault && fun_slip) {
        (*fun_slip)(fctNo, f_q, false);
        return true;
    }
    if (bc == BC::Dirichlet && fun_dirichlet) {
        (*fun_dirichlet)(fctNo, f_q, false);
        return true;
    }
    return false;
}

bool Viscoelasticity::bc_boundary(std::size_t fctNo, BC bc, double f_q_raw[]) const {
    auto f_q = Matrix<double>(f_q_raw, NumQuantities, fctRule.size());
    if (bc == BC::Fault && fun_slip) {
        (*fun_slip)(fctNo, f_q, true);
        // Fault slip is halved at boundary facets (one-sided average)
        for (std::size_t q = 0; q < fctRule.size(); ++q) {
            for (std::size_t p = 0; p < NumQuantities; ++p) {
                f_q(p, q) *= 0.5;
            }
        }
        return true;
    }
    if (bc == BC::Dirichlet && fun_dirichlet) {
        (*fun_dirichlet)(fctNo, f_q, true);
        return true;
    }
    return false;
}

bool Viscoelasticity::bc_traction(std::size_t fctNo, BC bc, double f_q_raw[]) const {
    if (bc != BC::Traction || !fun_traction)
        return false;
    auto f_q = Matrix<double>(f_q_raw, NumQuantities, fctRule.size());
    (*fun_traction)(fctNo, f_q, true);
    return true;
}

bool Viscoelasticity::bc_free_slip(std::size_t fctNo, BC bc, double f_q_raw[]) const {
    if (bc != BC::FreeSlip || !fun_free_slip)
        return false;
    auto f_q = Matrix<double>(f_q_raw, 1, fctRule.size());
    (*fun_free_slip)(fctNo, f_q, true);
    return true;
}

// RHS assembly — standard terms (body force, Dirichlet/fault, traction, slip)
bool Viscoelasticity::rhs_volume(std::size_t elNo, Vector<double>& B,
                                 LinearAllocator<double>& scratch) const {
    // Body force term: ∫_E f_i v_i |J| W dξ
    if (fun_force) {
        double F_Q_raw[tensor::F_Q::size()];
        auto F_Q = Matrix<double>(F_Q_raw, NumQuantities, volRule.size());
        (*fun_force)(elNo, F_Q);

        kernel::rhsVolumeForce rhsF;
        rhsF.b = B.data();
        rhsF.E_Q = E_Q.data();
        rhsF.F_Q = F_Q_raw;
        rhsF.J = vol[elNo].get<AbsDetJ>().data();
        rhsF.W = volRule.weights().data();
        rhsF.execute();
    }

    // History source term: −∫_E ε(v):σ̂ⁿ dE (uses OLD ε^dev and q from strainHistory_Q)
    alignas(ALIGNMENT) double Dx_Q[tensor::Dx_Q::size()];
    {
        kernel::Dx_Q dxKrnl;
        dxKrnl.Dx_Q = Dx_Q;
        dxKrnl.Dxi_Q = Dxi_Q.data();
        dxKrnl.G = vol[elNo].get<JInv>().data()->data();
        dxKrnl.execute();
    }

    kernel::rhsVolumeHistory rhsH;
    rhsH.b = B.data();
    rhsH.Dx_Q = Dx_Q;
    rhsH.mu1_W_J_Q = volPre[elNo].get<mu1_W_J_Q>().data();
    rhsH.g_dt_Q = volPre[elNo].get<g_dt_Q>().data();
    rhsH.ratio_Q = volPre[elNo].get<ratio_Q>().data();
    // Uses the OLD strain fields (time n); update happens in update_internal_state
    rhsH.partial_strain_tensor_Q = strainHistory_Q[elNo].get<partial_strain_old_Q>().data()->data();
    rhsH.deviatoric_strain_tensor_Q =
        strainHistory_Q[elNo].get<deviatoric_strain_old_Q>().data()->data();
    rhsH.execute();
    return true;
}

// Shared RHS lift setup used by rhs_skeleton and rhs_boundary.
// Fills f_lifted_q from f_q_raw using either BR2 or IP method.
void Viscoelasticity::compute_rhs_lift(std::size_t fctNo, FacetInfo const& info, bool is_skeleton,
                                       const double* f_q_raw, double* f_lifted_q) const {
    if (method_ == DGMethod::BR2) {
        if (is_skeleton) {
            alignas(ALIGNMENT) double f_lifted0[tensor::f_lifted::size(0)];
            alignas(ALIGNMENT) double f_lifted1[tensor::f_lifted::size(1)];
            alignas(ALIGNMENT) double Minv[2][tensor::M::size()];
            for (int i = 0; i < 2; ++i)
                compute_inverse_mass_matrix(info.up[i], Minv[i]);

            kernel::rhs_lift_skeleton lift;
            lift.delta = init::delta::Values;
            lift.f_q = f_q_raw;
            lift.f_lifted(0) = f_lifted0;
            lift.f_lifted(1) = f_lifted1;
            lift.f_lifted_q = f_lifted_q;
            lift.n_q = fct[fctNo].get<Normal>().data()->data();
            lift.w = fctRule.weights().data();
            for (int i = 0; i < 2; ++i) {
                lift.E_q(i) = E_q[info.localNo[i]].data();
                lift.Minv(i) = Minv[i];
                lift.A_dt_q(i) = (i == 0) ? fctPre[fctNo].get<A_dt_q_0>().data()
                                          : fctPre[fctNo].get<A_dt_q_1>().data();
                lift.B_dt_q(i) = (i == 0) ? fctPre[fctNo].get<B_dt_q_0>().data()
                                          : fctPre[fctNo].get<B_dt_q_1>().data();
            }
            lift.execute();
        } else {
            alignas(ALIGNMENT) double f_lifted0[tensor::f_lifted::size(0)];
            alignas(ALIGNMENT) double Minv0[tensor::M::size()];
            compute_inverse_mass_matrix(info.up[0], Minv0);

            kernel::rhs_lift_boundary lift;
            lift.delta = init::delta::Values;
            lift.f_q = f_q_raw;
            lift.f_lifted(0) = f_lifted0;
            lift.f_lifted_q = f_lifted_q;
            lift.n_q = fct[fctNo].get<Normal>().data()->data();
            lift.w = fctRule.weights().data();
            lift.E_q(0) = E_q[info.localNo[0]].data();
            lift.Minv(0) = Minv0;
            lift.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
            lift.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
            lift.execute();
        }
    } else { // IP
        kernel::rhs_lift_ip lift;
        lift.f_q = f_q_raw;
        lift.f_lifted_q = f_lifted_q;
        lift.nl_q = fct[fctNo].get<NormalLength>().data();
        lift.execute();
    }
}

bool Viscoelasticity::rhs_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                                   Vector<double>& B1, LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    if (!bc_skeleton(fctNo, info.bc, f_q_raw))
        return false;

    alignas(ALIGNMENT) double f_lifted_q[tensor::f_lifted_q::size()];
    compute_rhs_lift(fctNo, info, /*is_skeleton=*/true, f_q_raw, f_lifted_q);

    // Side 0 contribution: c1 = ε/2 (symmetry), c2 = γ (penalty)
    alignas(ALIGNMENT) double Dx_q[tensor::Dx_q::size(0)];
    kernel::rhsFacet rhs;
    rhs.f_q = f_q_raw;
    rhs.f_lifted_q = f_lifted_q;
    rhs.n_q = fct[fctNo].get<Normal>().data()->data();
    rhs.w = fctRule.weights().data();

    rhs.b = B0.data();
    rhs.c10 = 0.5 * epsilon;
    rhs.c20 = penalty(fctNo);
    rhs.Dx_q(0) = Dx_q;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
    rhs.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
    rhs.execute();

    // Side 1 contribution: flip signs of penalty
    rhs.b = B1.data();
    rhs.c20 = -penalty(fctNo);
    rhs.Dxi_q(0) = Dxi_q[info.localNo[1]].data();
    rhs.E_q(0) = E_q[info.localNo[1]].data();
    rhs.g(0) = fct[fctNo].get<JInv1>().data()->data();
    rhs.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_1>().data();
    rhs.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_1>().data();
    rhs.execute();
    return true;
}

bool Viscoelasticity::rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                                   LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    if (!bc_boundary(fctNo, info.bc, f_q_raw))
        return false;

    alignas(ALIGNMENT) double f_lifted_q[tensor::f_lifted_q::size()];
    compute_rhs_lift(fctNo, info, /*is_skeleton=*/false, f_q_raw, f_lifted_q);

    alignas(ALIGNMENT) double Dx_q[tensor::Dx_q::size(0)];
    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = epsilon;
    rhs.c20 = penalty(fctNo);
    rhs.f_q = f_q_raw;
    rhs.f_lifted_q = f_lifted_q;
    rhs.n_q = fct[fctNo].get<Normal>().data()->data();
    rhs.w = fctRule.weights().data();
    rhs.Dx_q(0) = Dx_q;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
    rhs.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
    rhs.execute();
    return true;
}

bool Viscoelasticity::rhs_boundary_free_slip(std::size_t fctNo, FacetInfo const& info,
                                             Vector<double>& B0,
                                             LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double f_q_raw[tensor::g_q::size()];
    if (!bc_free_slip(fctNo, info.bc, f_q_raw))
        return false;

    alignas(ALIGNMENT) double Dx_q[tensor::Dx_q::size(0)];
    kernel::rhsFreeSlip rhs;
    rhs.b = B0.data();
    rhs.c10 = epsilon;
    rhs.c20 = penalty(fctNo);
    rhs.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    rhs.nl_q = fct[fctNo].get<NormalLength>().data();
    rhs.w = fctRule.weights().data();
    rhs.g_q = f_q_raw;
    rhs.Dx_q(0) = Dx_q;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
    rhs.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
    rhs.execute();
    return true;
}

bool Viscoelasticity::rhs_traction_boundary(std::size_t fctNo, FacetInfo const& info,
                                            Vector<double>& B0,
                                            LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    if (!bc_traction(fctNo, info.bc, f_q_raw))
        return false;

    kernel::rhsTraction rhs;
    rhs.b = B0.data();
    rhs.w = fctRule.weights().data();
    rhs.nl_q = fct[fctNo].get<NormalLength>().data();
    rhs.traction_component = f_q_raw;
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.execute();

    return true;
}

/*
RHS history terms — ∫_∂E [[v]] {σ̂ⁿ n} dS

These are the NEW VE-specific surface terms. They use the OLD (time n)
strain fields read from strainHistory_q.

The tractionHistoryAverage kernel computes {σ̂ⁿ_pj n_j} = average of the
history traction across both sides. c3=2 accounts for the factor of 2
in σ̂ⁿ = 2μ₁[...].

Boundary facets pass side-0 history data into both tractionHistoryAverage
inputs because only one interior side exists.

Helper: populate the tractionHistoryAverage kernel for a facet with the
given history field pointers for each side.
*/
void Viscoelasticity::compute_traction_history_average(std::size_t fctNo, const double* dev_q0,
                                                       const double* partial_q0,
                                                       const double* mu1_q0, const double* dev_q1,
                                                       const double* partial_q1,
                                                       const double* mu1_q1) const {

    kernel::tractionHistoryAverage krnl;
    krnl.g_dt_q = fctPre[fctNo].get<g_dt_q>().data();
    krnl.ratio_q = fctPre[fctNo].get<ratio_q>().data();
    krnl.n_q = fct[fctNo].get<Normal>().data()->data();
    krnl.tractionHistoryAverage = strainHistory_q[fctNo].get<average_traction>().data()->data();
    krnl.deviatoric_strain_tensor_q(0) = dev_q0;
    krnl.partial_strain_tensor_q(0) = partial_q0;
    krnl.mu1_q(0) = mu1_q0;
    krnl.deviatoric_strain_tensor_q(1) = dev_q1;
    krnl.partial_strain_tensor_q(1) = partial_q1;
    krnl.mu1_q(1) = mu1_q1;
    krnl.execute();
}

bool Viscoelasticity::rhs_history_skeleton(std::size_t fctNo, FacetInfo const& info,
                                           Vector<double>& B0, Vector<double>& B1,
                                           LinearAllocator<double>& scratch) const {
    // Interior facet: genuine two-sided average of σ̂ⁿ
    compute_traction_history_average(
        fctNo, strainHistory_q[fctNo].get<deviatoric_strain_old_q_0>().data()->data(),
        strainHistory_q[fctNo].get<partial_strain_old_q_0>().data()->data(),
        fctPre[fctNo].get<mu1_q_0>().data(),
        strainHistory_q[fctNo].get<deviatoric_strain_old_q_1>().data()->data(),
        strainHistory_q[fctNo].get<partial_strain_old_q_1>().data()->data(),
        fctPre[fctNo].get<mu1_q_1>().data());

    const double* avg = strainHistory_q[fctNo].get<average_traction>().data()->data();

    // [[v]] = v⁻ − v⁺: side 0 gets +c3, side 1 gets −c3
    kernel::rhsSurface rhs;
    rhs.tractionHistoryAverage = avg;
    rhs.w = fctRule.weights().data();

    rhs.b = B0.data();
    rhs.c3 = 2.0;
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.execute();

    rhs.b = B1.data();
    rhs.c3 = -2.0;
    rhs.E_q(0) = E_q[info.localNo[1]].data();
    rhs.execute();
    return true;
}

bool Viscoelasticity::rhs_history_boundary(std::size_t fctNo, FacetInfo const& info,
                                           Vector<double>& B0,
                                           LinearAllocator<double>& scratch) const {
    if (info.bc == BC::Natural || info.bc == BC::Traction || info.bc == BC::FreeSlip) {
        return false;
    }

    // Boundary has only one interior side; supply side-0 data to both kernel slots.
    const double* dev = strainHistory_q[fctNo].get<deviatoric_strain_old_q_0>().data()->data();
    const double* partial = strainHistory_q[fctNo].get<partial_strain_old_q_0>().data()->data();
    const double* mu1 = fctPre[fctNo].get<mu1_q_0>().data();

    compute_traction_history_average(fctNo, dev, partial, mu1, dev, partial, mu1);

    kernel::rhsSurface rhs;
    rhs.tractionHistoryAverage = strainHistory_q[fctNo].get<average_traction>().data()->data();
    rhs.c3 = 2.0;
    rhs.b = B0.data();
    rhs.w = fctRule.weights().data();
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.execute();
    return true;
}

bool Viscoelasticity::rhs_history_boundary_for_free_slip(std::size_t fctNo, FacetInfo const& info,
                                                         Vector<double>& B0,
                                                         LinearAllocator<double>& scratch) const {
    if (info.bc != BC::FreeSlip)
        return false;

    // Same one-sided pattern as rhs_history_boundary
    const double* dev = strainHistory_q[fctNo].get<deviatoric_strain_old_q_0>().data()->data();
    const double* partial = strainHistory_q[fctNo].get<partial_strain_old_q_0>().data()->data();
    const double* mu1 = fctPre[fctNo].get<mu1_q_0>().data();
    compute_traction_history_average(fctNo, dev, partial, mu1, dev, partial, mu1);

    // For free-slip, only the normal component of the history traction
    // contributes (tangential components are unconstrained).
    kernel::rhsSurfaceNormal rhs;
    rhs.tractionHistoryAverage = strainHistory_q[fctNo].get<average_traction>().data()->data();
    rhs.tractionHistoryNormal = strainHistory_q[fctNo].get<traction_history_normal>().data();
    rhs.c3 = 2.0;
    rhs.b = B0.data();
    rhs.w = fctRule.weights().data();
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    rhs.execute();
    return true;
}

bool Viscoelasticity::rhs_free_slip_boundary(std::size_t fctNo, FacetInfo const& info,
                                             Vector<double>& B0,
                                             LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double g_q_raw[tensor::g_q::size()];
    if (!bc_free_slip(fctNo, info.bc, g_q_raw))
        return false;

    alignas(ALIGNMENT) double Dx_q[tensor::Dx_q::size(0)];
    kernel::rhsFreeSlip rhs;
    rhs.b = B0.data();
    rhs.c10 = epsilon;
    rhs.c20 = penalty(fctNo);
    rhs.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    rhs.nl_q = fct[fctNo].get<NormalLength>().data();
    rhs.w = fctRule.weights().data();
    rhs.g_q = g_q_raw;
    rhs.Dx_q(0) = Dx_q;
    rhs.Dxi_q(0) = Dxi_q[info.localNo[0]].data();
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
    rhs.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
    rhs.execute();
    return true;
}

} // namespace tndm
