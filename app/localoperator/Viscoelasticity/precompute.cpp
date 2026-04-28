/*
Precomputation for Viscoelasticity local operator

Volume and facet precomputation: material L2 projection, folded W*J arrays,
time-dependent A(Δt)/B(Δt) updates, and skeleton/boundary preparation.
*/

#include "Viscoelasticity.h"

#include "kernels/viscoelasticity/init.h"
#include "kernels/viscoelasticity/kernel.h"
#include "kernels/viscoelasticity/tensor.h"
#include "tensor/EigenMap.h"
#include "util/LinearAllocator.h"

#include <Eigen/LU>
#include <cmath>

namespace tensor = tndm::viscoelasticity::tensor;
namespace init = tndm::viscoelasticity::init;
namespace kernel = tndm::viscoelasticity::kernel;

namespace tndm {

// Precomputation — volume
//
// prepare_volume: material L2 projection (time-independent, called once)
// prepare_volume_post_skeleton: folded W*J arrays + A_dt/B_dt (time-dependent,
//   called each step because g(Δt) and exp(-Δt/τ) change with dt)
//
// g_dt and ratio are computed per quad point from the local relaxation_time_local field.
// This handles spatially varying viscosity.

void Viscoelasticity::prepare_volume(std::size_t elNo, LinearAllocator<double>& scratch) {
    base::prepare_volume(elNo, scratch);

    // Evaluate all material functionals at volume quad points
    alignas(ALIGNMENT) double lam_Q_raw[tensor::lam_Q::size()];
    alignas(ALIGNMENT) double mu0_Q_raw[tensor::mu0_Q::size()];
    alignas(ALIGNMENT) double mu1_Q_raw[tensor::mu1_Q::size()];
    alignas(ALIGNMENT) double viscosity_Q_raw[tensor::viscosity_Q::size()];
    alignas(ALIGNMENT) double rhoInv_Q_raw[tensor::rhoInv_Q::size()];

    // Wrap raw arrays in Matrix views (must be lvalues for functional interface)
    auto lam_Q = Matrix<double>(lam_Q_raw, 1, volRule.size());
    auto mu0_Q = Matrix<double>(mu0_Q_raw, 1, volRule.size());
    auto mu1_Q = Matrix<double>(mu1_Q_raw, 1, volRule.size());
    auto viscosity_Q = Matrix<double>(viscosity_Q_raw, 1, volRule.size());
    auto rhoInv_Q = Matrix<double>(rhoInv_Q_raw, 1, volRule.size());

    fun_lam(elNo, lam_Q);
    fun_mu0(elNo, mu0_Q);
    fun_mu1(elNo, mu1_Q);
    fun_viscosity(elNo, viscosity_Q);
    fun_rho(elNo, rhoInv_Q);

    // Invert density in-place: rhoInv_Q = 1/ρ(x_q)
    for (unsigned q = 0; q < tensor::rhoInv_Q::Shape[0]; ++q) {
        rhoInv_Q_raw[q] = 1.0 / rhoInv_Q_raw[q];
    }

    // L2 projection: quad-point values → nodal basis coefficients
    alignas(ALIGNMENT) double Mmem[tensor::matM::size()];
    kernel::project_material_lhs krnl_lhs;
    krnl_lhs.matE_Q_T = matE_Q_T.data();
    krnl_lhs.J = vol[elNo].get<AbsDetJ>().data();
    krnl_lhs.matM = Mmem;
    krnl_lhs.W = volRule.weights().data();
    krnl_lhs.execute();

    kernel::project_material_rhs krnl_rhs;
    krnl_rhs.matE_Q_T = matE_Q_T.data();
    krnl_rhs.J = vol[elNo].get<AbsDetJ>().data();
    krnl_rhs.W = volRule.weights().data();
    krnl_rhs.lam = material[elNo].get<lam>().data();
    krnl_rhs.lam_Q = lam_Q_raw;
    krnl_rhs.mu0 = material[elNo].get<mu0>().data();
    krnl_rhs.mu0_Q = mu0_Q_raw;
    krnl_rhs.mu1 = material[elNo].get<mu1>().data();
    krnl_rhs.mu1_Q = mu1_Q_raw;
    krnl_rhs.viscosity = material[elNo].get<viscosity>().data();
    krnl_rhs.viscosity_Q = viscosity_Q_raw;
    krnl_rhs.rhoInv = material[elNo].get<rhoInv>().data();
    krnl_rhs.rhoInv_Q = rhoInv_Q_raw;
    krnl_rhs.execute();

    // Solve the mass system for each material field (nodal L2 projection solve)
    using MMap = Eigen::Map<Eigen::Matrix<double, tensor::matM::Shape[0], tensor::matM::Shape[1]>,
                            Eigen::Unaligned,
                            Eigen::OuterStride<init::matM::Stop[0] - init::matM::Start[0]>>;
    auto proj = MMap(Mmem).fullPivLu();

    using Vec1 = Eigen::Map<Eigen::Matrix<double, tensor::lam::Shape[0], 1>, Eigen::Unaligned,
                            Eigen::InnerStride<1>>;
    Vec1(material[elNo].get<lam>().data()) = proj.solve(Vec1(material[elNo].get<lam>().data()));
    Vec1(material[elNo].get<rhoInv>().data()) =
        proj.solve(Vec1(material[elNo].get<rhoInv>().data()));
    Vec1(material[elNo].get<mu0>().data()) = proj.solve(Vec1(material[elNo].get<mu0>().data()));
    Vec1(material[elNo].get<mu1>().data()) = proj.solve(Vec1(material[elNo].get<mu1>().data()));
    Vec1(material[elNo].get<viscosity>().data()) =
        proj.solve(Vec1(material[elNo].get<viscosity>().data()));
}

void Viscoelasticity::prepare_volume_post_skeleton(std::size_t elNo,
                                                   LinearAllocator<double>& scratch) {
    base::prepare_volume_post_skeleton(elNo, scratch);

    // Compute Jinv_Q = 1/|J| for the rhoInv term in wave_rhs
    auto J_Q = vol[elNo].get<AbsDetJ>();
    alignas(ALIGNMENT) double Jinv_Q[tensor::Jinv_Q::size()] = {};
    for (unsigned q = 0; q < tensor::Jinv_Q::Shape[0]; ++q) {
        Jinv_Q[q] = 1.0 / J_Q[q];
    }

    // Time-independent folded arrays: lam*W*J, mu0*W*J, mu1*W*J, etc.
    kernel::precomputeVolume krnl_pre;
    krnl_pre.matE_Q_T = matE_Q_T.data();
    krnl_pre.J = vol[elNo].get<AbsDetJ>().data();
    krnl_pre.Jinv_Q = Jinv_Q;
    krnl_pre.W = volRule.weights().data();
    krnl_pre.lam = material[elNo].get<lam>().data();
    krnl_pre.lam_W_J_Q = volPre[elNo].template get<lam_W_J_Q>().data();
    krnl_pre.mu0 = material[elNo].get<mu0>().data();
    krnl_pre.mu0_W_J_Q = volPre[elNo].template get<mu0_W_J_Q>().data();
    krnl_pre.mu1 = material[elNo].get<mu1>().data();
    krnl_pre.mu1_W_J_Q = volPre[elNo].template get<mu1_W_J_Q>().data();
    krnl_pre.viscosity = material[elNo].get<viscosity>().data();
    krnl_pre.viscosity_W_J_Q = volPre[elNo].template get<viscosity_W_J_Q>().data();
    krnl_pre.rhoInv = material[elNo].get<rhoInv>().data();
    krnl_pre.negative_rhoInv_W_Jinv_Q =
        volPre[elNo].template get<negative_rhoInv_W_Jinv_Q>().data();
    krnl_pre.execute();

    // Note: Time-dependent quantities (g_dt, ratio, A_dt, B_dt) are updated in
    // update_time_dependent_precomputation_volume(), which is called from DGOperator
    // after the viscoelastic time step is set.
}

// Precomputation — facets (skeleton and boundary)
//
// Facet g_dt and ratio use the same per-quad-point treatment as in volume precomputation.
// A shared helper provides common setup for both prepare_skeleton and prepare_boundary.

// Shared setup for both skeleton and boundary facets (side 0 always present,
// side 1 only present for skeleton — caller is responsible for which sides).
void Viscoelasticity::precompute_facet_material(std::size_t fctNo, FacetInfo const& info,
                                                int numSides) {
    // Interpolate nodal material values to facet quad points (time-independent)
    kernel::precomputeSurface krnl;
    for (int side = 0; side < numSides; ++side) {
        krnl.lam = material[info.up[side]].get<lam>().data();
        krnl.mu0 = material[info.up[side]].get<mu0>().data();
        krnl.mu1 = material[info.up[side]].get<mu1>().data();
        krnl.viscosity = material[info.up[side]].get<viscosity>().data();
        krnl.matE_q_T(side) = matE_q_T[info.localNo[side]].data();

        if (side == 0) {
            krnl.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
            krnl.mu0_q(0) = fctPre[fctNo].get<mu0_q_0>().data();
            krnl.mu1_q(0) = fctPre[fctNo].get<mu1_q_0>().data();
            krnl.viscosity_q(0) = fctPre[fctNo].get<viscosity_q_0>().data();
        } else {
            krnl.lam_q(1) = fctPre[fctNo].get<lam_q_1>().data();
            krnl.mu0_q(1) = fctPre[fctNo].get<mu0_q_1>().data();
            krnl.mu1_q(1) = fctPre[fctNo].get<mu1_q_1>().data();
            krnl.viscosity_q(1) = fctPre[fctNo].get<viscosity_q_1>().data();
        }
        krnl.execute(side);
    }

    // Note: Time-dependent quantities (g_dt_q, ratio_q, A_dt_q, B_dt_q) are updated
    // in update_time_dependent_precomputation_boundary/skeleton(), called from DGOperator.
}

void Viscoelasticity::prepare_skeleton(std::size_t fctNo, FacetInfo const& info,
                                       LinearAllocator<double>& scratch) {
    base::prepare_skeleton(fctNo, info, scratch);
    precompute_facet_material(fctNo, info, 2);
    transpose_JInv(fctNo, 0);
    transpose_JInv(fctNo, 1);
}

void Viscoelasticity::prepare_boundary(std::size_t fctNo, FacetInfo const& info,
                                       LinearAllocator<double>& scratch) {
    base::prepare_boundary(fctNo, info, scratch);
    precompute_facet_material(fctNo, info, 1);
    transpose_JInv(fctNo, 0);
}

// Time-dependent precomputation updates
//
// This method recomputes g(Δt), ratio=exp(-Δt/τ), A(Δt), and B(Δt) for all
// volume and facet elements when called from DGOperator::assemble().
// This ensures time-dependent quantities are always current before assembly.

void Viscoelasticity::update_time_dependent_precomputation_volume(std::size_t elNo) {
    auto g_dt_data = volPre[elNo].get<g_dt_Q>().data();
    auto ratio_data = volPre[elNo].get<ratio_Q>().data();
    auto tau_field = volPre[elNo].get<relaxation_time_local>().data();

    for (std::size_t q = 0; q < volRule.size(); ++q) {
        const double tau_q = tau_field[q];
        g_dt_data[q] = compute_g_dt(dt_viscoelastic_, tau_q, 200);
        ratio_data[q] = (tau_q > 0.0) ? std::exp(-dt_viscoelastic_ / tau_q) : 0.0;
    }

    kernel::precomputeVolumeAB krnl_pre_ab;
    krnl_pre_ab.lam_W_J_Q = volPre[elNo].get<lam_W_J_Q>().data();
    krnl_pre_ab.mu0_W_J_Q = volPre[elNo].get<mu0_W_J_Q>().data();
    krnl_pre_ab.mu1_W_J_Q = volPre[elNo].get<mu1_W_J_Q>().data();
    krnl_pre_ab.A_dt = volPre[elNo].get<A_dt>().data();
    krnl_pre_ab.B_dt = volPre[elNo].get<B_dt>().data();
    krnl_pre_ab.g_dt_Q = volPre[elNo].get<g_dt_Q>().data();
    krnl_pre_ab.execute();
}

void Viscoelasticity::update_time_dependent_precomputation_surface(std::size_t fctNo,
                                                                   int numSides) {
    auto g_dt_data = fctPre[fctNo].get<g_dt_q>().data();
    auto ratio_data = fctPre[fctNo].get<ratio_q>().data();
    auto mu1_q0 = fctPre[fctNo].get<mu1_q_0>().data();
    auto viscosity_q0 = fctPre[fctNo].get<viscosity_q_0>().data();

    for (std::size_t q = 0; q < fctRule.size(); ++q) {
        const double tau_q = (mu1_q0[q] != 0.0) ? viscosity_q0[q] / mu1_q0[q] : 0.0;
        g_dt_data[q] = compute_g_dt(dt_viscoelastic_, tau_q, 200);
        ratio_data[q] = (tau_q > 0.0) ? std::exp(-dt_viscoelastic_ / tau_q) : 0.0;
    }

    kernel::precomputeSurfaceAB krnl_ab;
    krnl_ab.g_dt_q = fctPre[fctNo].get<g_dt_q>().data();
    krnl_ab.lam_q(0) = fctPre[fctNo].get<lam_q_0>().data();
    krnl_ab.mu0_q(0) = fctPre[fctNo].get<mu0_q_0>().data();
    krnl_ab.mu1_q(0) = fctPre[fctNo].get<mu1_q_0>().data();
    krnl_ab.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
    krnl_ab.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
    krnl_ab.execute(0);

    if (numSides == 2) {
        krnl_ab.lam_q(1) = fctPre[fctNo].get<lam_q_1>().data();
        krnl_ab.mu0_q(1) = fctPre[fctNo].get<mu0_q_1>().data();
        krnl_ab.mu1_q(1) = fctPre[fctNo].get<mu1_q_1>().data();
        krnl_ab.A_dt_q(1) = fctPre[fctNo].get<A_dt_q_1>().data();
        krnl_ab.B_dt_q(1) = fctPre[fctNo].get<B_dt_q_1>().data();
        krnl_ab.execute(1);
    }
}

void Viscoelasticity::update_time_dependent_precomputation_skeleton(std::size_t fctNo) {
    update_time_dependent_precomputation_surface(fctNo, 2);
}

void Viscoelasticity::update_time_dependent_precomputation_boundary(std::size_t fctNo) {
    update_time_dependent_precomputation_surface(fctNo, 1);
}

} // namespace tndm
