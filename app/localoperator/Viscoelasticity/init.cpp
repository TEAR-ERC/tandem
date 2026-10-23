/*
Viscoelasticity initialization operations

Constructor, g(Δt) computation, mass matrix utilities, memory allocation,
relaxation time, Jacobian transpose, stiffness/penalty/CFL helpers, and
lift operator helpers shared by assembly and RHS.
*/

#include "Viscoelasticity.h"

#include "basis/Equidistant.h"
#include "basis/WarpAndBlend.h"
#include "config.h"
#include "form/BC.h"
#include "form/DGCurvilinearCommon.h"
#include "form/InverseInequality.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "kernels/viscoelasticity/init.h"
#include "kernels/viscoelasticity/kernel.h"
#include "kernels/viscoelasticity/tensor.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "tensor/EigenMap.h"
#include "util/LinearAllocator.h"

#include <Eigen/LU>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tensor = tndm::viscoelasticity::tensor;
namespace init = tndm::viscoelasticity::init;
namespace kernel = tndm::viscoelasticity::kernel;

namespace tndm {

// Constructor
Viscoelasticity::Viscoelasticity(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                                 functional_t_region<1> lam, functional_t_region<1> mu0,
                                 functional_t_region<1> mu1, functional_t_region<1> viscosity,
                                 std::optional<functional_t_region<1>> relaxation_time,
                                 double theta, std::optional<functional_t_region<1>> rho,
                                 DGMethod method)
    : DGCurvilinearCommon<DomainDimension>(std::move(cl), MinQuadOrder()), method_(method),
      space_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>(), ALIGNMENT),
      materialSpace_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>(), ALIGNMENT),
      fun_lam(make_volume_functional(std::move(lam))),
      fun_mu0(make_volume_functional(std::move(mu0))),
      fun_mu1(make_volume_functional(std::move(mu1))),
      fun_viscosity(make_volume_functional(std::move(viscosity))),
      fun_relaxation_time(
          relaxation_time ? std::make_optional(make_volume_functional(std::move(*relaxation_time)))
                          : std::nullopt),
      theta_(theta), fun_rho(rho ? make_volume_functional(std::move(*rho)) : one_volume_function) {

    // theta in (0,1]: fraction of relaxation time used as time-step upper bound.
    // 0 would freeze time; >1 would over-step the relaxation.
    if (theta_ <= 0.0 || theta_ > 1.0) {
        throw std::invalid_argument("VE: theta must be in (0, 1]");
    }

    // Basis function evaluations — precomputed once at construction
    MhatInv = space_.inverseMassMatrix();
    E_Q = space_.evaluateBasisAt(volRule.points());
    E_Q_T = space_.evaluateBasisAt(volRule.points(), {1, 0});
    Dxi_Q = space_.evaluateGradientAt(volRule.points());
    Dxi_Q_120 = space_.evaluateGradientAt(volRule.points(), {1, 2, 0});

    auto refNodes = EquidistantNodesFactory<Dim>(NumberingConvention::VTK)(PolynomialDegree);
    Dxi = space_.evaluateGradientAt(refNodes, {2, 0, 1});

    negative_E_Q_T = Managed<Matrix<double>>(E_Q_T.shape(), std::size_t{ALIGNMENT});
    EigenMap(negative_E_Q_T) = -EigenMap(E_Q_T);

    matE_Q_T = materialSpace_.evaluateBasisAt(volRule.points(), {1, 0});

    // Facet basis evaluations for each local facet reference map
    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto points = cl_->facetParam(f, fctRule.points());
        E_q.emplace_back(space_.evaluateBasisAt(points));
        E_q_T.emplace_back(space_.evaluateBasisAt(points, {1, 0}));
        Dxi_q.emplace_back(space_.evaluateGradientAt(points));
        Dxi_q_120.emplace_back(space_.evaluateGradientAt(points, {1, 2, 0}));
        matE_q_T.emplace_back(materialSpace_.evaluateBasisAt(points, {1, 0}));

        negative_E_q.emplace_back(space_.evaluateBasisAt(points));
        EigenMap(negative_E_q.back()) = -EigenMap(negative_E_q.back());

        negative_E_q_T.emplace_back(space_.evaluateBasisAt(points, {1, 0}));
        EigenMap(negative_E_q_T.back()) = -EigenMap(negative_E_q_T.back());
    }
}
/*
g(Δt) — closed-form time integration function

g(Δt) = (τ/Δt)(1 − exp(−Δt/τ)) = (1 − exp(−r))/r,  r = Δt/τ

Uses expm1 for stable evaluation when r is small.
*/
double Viscoelasticity::compute_g_dt(double dt, double tau) const {
    if (dt <= 0.0 || tau <= 0.0) {
        return 0.0;
    }
    double const r = dt / tau;
    if (r == 0.0) { // r underflowed; g(0) = 1
        return 1.0;
    }
    // -expm1(-r) == 1 - exp(-r) evaluated without the cancellation that the naive
    // form suffers as r -> 0 (the coseismic regime, where r ~ 1e-12).
    return -std::expm1(-r) / r;
}

// Mass matrix utilities
void Viscoelasticity::compute_mass_matrix(std::size_t elNo, double* M) const {
    kernel::massMatrix mm;
    mm.E_Q = E_Q.data();
    mm.J = vol[elNo].get<AbsDetJ>().data();
    mm.M = M;
    mm.W = volRule.weights().data();
    mm.execute();
}

void Viscoelasticity::compute_inverse_mass_matrix(std::size_t elNo, double* Minv) const {
    compute_mass_matrix(elNo, Minv);

    using MMat = Eigen::Matrix<double, tensor::M::Shape[0], tensor::M::Shape[1]>;
    using MMap = Eigen::Map<MMat, Eigen::Unaligned,
                            Eigen::OuterStride<init::M::Stop[0] - init::M::Start[0]>>;
    auto m = MMap(Minv);
    m = Eigen::FullPivLU<MMat>(m).inverse();
}

// Memory allocation for all viscoelastic Mneme containers
void Viscoelasticity::begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                                        std::size_t numLocalFacets) {
    base::begin_preparation(numElements, numLocalElements, numLocalFacets);

    material.setStorage(
        std::make_shared<material_vol_t>(numElements * materialSpace_.numBasisFunctions()), 0u,
        numElements, materialSpace_.numBasisFunctions());

    volPre.setStorage(std::make_shared<vol_pre_t>(numElements * volRule.size()), 0u, numElements,
                      volRule.size());

    fctPre.setStorage(std::make_shared<fct_pre_t>(numLocalFacets * fctRule.size()), 0u,
                      numLocalFacets, fctRule.size());

    strainHistory_Q.setStorage(std::make_shared<strain_tensor_Q_t>(numElements * volRule.size()),
                               0u, numElements, volRule.size());

    strainHistory_q.setStorage(std::make_shared<strain_tensor_q_t>(numLocalFacets * fctRule.size()),
                               0u, numLocalFacets, fctRule.size());

    displacementField.setStorage(
        std::make_shared<displacement_field_t>(numElements * space_.numBasisFunctions()), 0u,
        numElements, space_.numBasisFunctions());

    penalty_.resize(numLocalFacets);
    cfl_dt_.resize(numLocalElements, 0.0);
}

/*
Relaxation time: range over the viscoelastic material

tau = viscosity / mu1, evaluated from the lua values at the volume quadrature
points. Feeds into DGOperator's global MPI reductions of the minimum and maximum
τ. The minimum constrains Δt ≤ θ·τ_min; the constitutive update uses the τ of each
quadrature point (see update_time_dependent_precomputation_volume).

Only points with μ₁ > 0 count: where the Maxwell branch is absent τ has no effect.
A point with μ₁ > 0 and viscosity ≤ 0 is kept, so that the global minimum becomes
≤ 0 and DGOperator rejects the input on every rank.

relaxation_time is deprecated. If the scenario still defines it, it must agree with
viscosity / mu1 wherever mu1 > 0; disagreeing points are counted and DGOperator
rejects the input.
*/
void Viscoelasticity::local_relaxation_time(std::size_t elNo, double& relaxation_time_global,
                                            LinearAllocator<double>& scratch) {
    base::prepare_volume(elNo, scratch);

    alignas(ALIGNMENT) double eta_raw[tensor::mu1_Q::size()];
    alignas(ALIGNMENT) double mu1_raw[tensor::mu1_Q::size()];
    alignas(ALIGNMENT) double given_raw[tensor::mu1_Q::size()];
    auto eta = Matrix<double>(eta_raw, 1, volRule.size());
    auto mu1 = Matrix<double>(mu1_raw, 1, volRule.size());
    auto given = Matrix<double>(given_raw, 1, volRule.size());
    fun_viscosity(elNo, eta);
    fun_mu1(elNo, mu1);
    if (fun_relaxation_time) {
        (*fun_relaxation_time)(elNo, given);
    }

    constexpr double rel_tol = 1.0e-10;
    for (std::size_t q = 0; q < volRule.size(); ++q) {
        if (mu1(0, q) > 0.0) {
            double const tau = eta(0, q) / mu1(0, q);
            relaxation_time_global = std::min(relaxation_time_global, tau);
            relaxation_time_max_local_ = std::max(relaxation_time_max_local_, tau);
            if (fun_relaxation_time && !(std::abs(given(0, q) - tau) <= rel_tol * std::abs(tau))) {
                ++relaxation_time_mismatch_local_;
            }
        }
    }
}

// tau = viscosity / mu1 at a quad point of the projected material. Where mu1 <= 0 there is
// no Maxwell branch and tau is infinite: g = 1 and ratio = 1, both multiplied by mu1.
double Viscoelasticity::relaxation_time_from(double viscosity, double mu1) {
    if (mu1 <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return viscosity / mu1;
}

void Viscoelasticity::set_relaxation_time_max_global(double relaxation_time) {
    // Sample points for viscoelastic_g_change: |g(dt/tau) - g(dt_g/tau)| is a smooth bump in
    // log(tau), so log-spaced samples between the extremes bound it closely.
    constexpr int num_samples = 33;
    double const tau_min = relaxation_time_global_;
    double const tau_max = std::max(relaxation_time, tau_min);
    tau_samples_.clear();
    if (!(tau_min > 0.0) || tau_max <= tau_min * (1.0 + 1.0e-12)) {
        tau_samples_.push_back(tau_min);
        return;
    }
    double const log_ratio = std::log(tau_max / tau_min);
    for (int i = 0; i < num_samples; ++i) {
        tau_samples_.push_back(tau_min * std::exp(log_ratio * i / (num_samples - 1)));
    }
    tau_samples_.back() = tau_max;
}

double Viscoelasticity::viscoelastic_g_change(double dt) const {
    double change = 0.0;
    if (tau_samples_.empty()) {
        return std::abs(compute_g_dt(dt, relaxation_time_global_) -
                        compute_g_dt(dt_g_, relaxation_time_global_));
    }
    for (double tau : tau_samples_) {
        change = std::max(change, std::abs(compute_g_dt(dt, tau) - compute_g_dt(dt_g_, tau)));
    }
    return change;
}

// exp(-dt/tau). tau <= 0 (only allowed where mu1 = 0) is the instant-relaxation limit 0,
// consistent with g(dt, tau <= 0) = 0.
double Viscoelasticity::compute_ratio(double dt, double tau) {
    if (dt <= 0.0) {
        return 1.0;
    }
    if (tau <= 0.0) {
        return 0.0;
    }
    return std::exp(-dt / tau);
}

// Jacobian transpose utility (needed for matrix-free apply, facet metrics)
void Viscoelasticity::transpose_JInv(std::size_t fctNo, int side) {
    const auto& G_q = (side == 1) ? fct[fctNo].get<JInv1>() : fct[fctNo].get<JInv0>();
    auto& G_q_T = (side == 1) ? fctPre[fctNo].get<JInvT1>() : fctPre[fctNo].get<JInvT0>();
    for (std::size_t q = 0; q < fctRule.size(); ++q) {
        for (std::ptrdiff_t i = 0; i < Dim; ++i) {
            for (std::ptrdiff_t j = 0; j < Dim; ++j) {
                G_q_T[q][i + j * Dim] = G_q[q][j + i * Dim];
            }
        }
    }
}

// Stiffness / penalty helpers
void Viscoelasticity::mu_avg(std::size_t fctNo, FacetInfo const& info,
                             Matrix<double>& result) const {
    // mu_avg is only valid for interior fault facets (two adjacent elements).
    // Boundary facets have no element on one side - info.up[1] would be invalid.
    assert(info.up[0] != info.up[1] && "mu_avg called on a boundary facet");
    double const* mu0_0 = material[info.up[0]].get<mu0>().data();
    double const* mu1_0 = material[info.up[0]].get<mu1>().data();
    double const* mu0_1 = material[info.up[1]].get<mu0>().data();
    double const* mu1_1 = material[info.up[1]].get<mu1>().data();
    double* res = result.data();
    std::size_t nq = result.size();

    #pragma omp simd
    for (std::size_t q = 0; q < nq; ++q) {
        res[q] = 0.5 * ((mu0_0[q] + mu1_0[q]) + (mu0_1[q] + mu1_1[q]));
    }
}

std::pair<double, double> Viscoelasticity::stiffness_tensor_bounds(std::size_t elNo) const {
    auto lam_field = material[elNo].get<lam>();
    auto mu0_field = material[elNo].get<mu0>();
    auto mu1_field = material[elNo].get<mu1>();
    assert(lam_field.size() == mu0_field.size());

    double c0 = std::numeric_limits<double>::max();
    double c1 = std::numeric_limits<double>::lowest();
    for (std::size_t i = 0, n = lam_field.size(); i < n; ++i) {
        const double mu_total = mu0_field[i] + mu1_field[i];
        c0 = std::min(c0, 2.0 * mu_total);
        c1 = std::max(c1, static_cast<double>(Dim) * lam_field[i] + 2.0 * mu_total);
    }
    return {c0, c1};
}

double Viscoelasticity::inverse_density_upper_bound(std::size_t elNo) const {
    auto rhoInv_field = material[elNo].get<rhoInv>();

    double ir1 = std::numeric_limits<double>::lowest();
    for (std::size_t i = 0, n = rhoInv_field.size(); i < n; ++i) {
        ir1 = std::max(ir1, rhoInv_field[i]);
    }
    return ir1;
}

void Viscoelasticity::prepare_penalty(std::size_t fctNo, FacetInfo const& info,
                                      LinearAllocator<double>&) {
    auto const p = [&](int side) {
        const auto [c0, c1] = stiffness_tensor_bounds(info.up[side]);
        constexpr double c_N_1 = InverseInequality<Dim>::trace_constant(PolynomialDegree - 1);
        return (Dim + 1) * c_N_1 * (area_[fctNo] / volume_[info.up[side]]) * (c1 * c1 / c0);
    };
    penalty_[fctNo] = (info.up[0] != info.up[1]) ? (p(0) + p(1)) / 4.0 : p(0);
}

void Viscoelasticity::prepare_cfl(std::size_t elNo, mneme::span<SideInfo> info,
                                  LinearAllocator<double>&) {
    double l_max = 0.0;
    double bnd_area = 0.0;
    for (std::size_t f = 0; f < NumFacets; ++f) {
        const bool is_skeleton = (elNo != info[f].lid);
        const double gamma = is_skeleton ? 2.0 : 1.0;
        const auto fctNo = info[f].fctNo;
        constexpr double c_N = InverseInequality<Dim>::trace_constant(PolynomialDegree);
        l_max += gamma * penalty_[fctNo] * c_N * area_[fctNo] / volume_[elNo];
        bnd_area += area_[fctNo];
    }
    const auto [c0, c1] = stiffness_tensor_bounds(elNo);
    const double ir1 = inverse_density_upper_bound(elNo);
    const double h_1 = bnd_area / volume_[elNo];
    constexpr double C_N = InverseInequality<Dim>::grad_constant(PolynomialDegree);
    l_max += c1 * ir1 * C_N * h_1 * h_1;
    l_max *= 2.0;
    cfl_dt_[elNo] = 1.0 / std::sqrt(l_max);
}

// Lift operator helpers — used by both assemble and rhs kernels.
void Viscoelasticity::compute_lift_boundary(std::size_t fctNo, FacetInfo const& info,
                                            double* L_q0) const {
    if (method_ == DGMethod::BR2) {
        alignas(ALIGNMENT) double Lift0[tensor::Lift::size(0)];
        alignas(ALIGNMENT) double Minv0[tensor::M::size()];
        compute_inverse_mass_matrix(info.up[0], Minv0);

        kernel::lift_boundary lift;
        lift.delta = init::delta::Values;
        lift.Lift(0) = Lift0;
        lift.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
        lift.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
        lift.n_q = fct[fctNo].get<Normal>().data()->data();
        lift.w = fctRule.weights().data();
        lift.E_q(0) = E_q[info.localNo[0]].data();
        lift.L_q(0) = L_q0;
        lift.Minv(0) = Minv0;
        lift.execute();
    } else {
        kernel::lift_ip lift;
        lift.delta = init::delta::Values;
        lift.nl_q = fct[fctNo].get<NormalLength>().data();
        lift.E_q(0) = E_q[info.localNo[0]].data();
        lift.L_q(0) = L_q0;
        lift.execute(0);
    }
}

void Viscoelasticity::compute_lift_skeleton(std::size_t fctNo, FacetInfo const& info, double* L_q0,
                                            double* L_q1) const {
    if (method_ == DGMethod::BR2) {
        alignas(ALIGNMENT) double Lift0[tensor::Lift::size(0)];
        alignas(ALIGNMENT) double Lift1[tensor::Lift::size(1)];
        alignas(ALIGNMENT) double Minv[2][tensor::M::size()];
        for (int i = 0; i < 2; ++i)
            compute_inverse_mass_matrix(info.up[i], Minv[i]);

        kernel::lift_skeleton lift;
        lift.delta = init::delta::Values;
        lift.Lift(0) = Lift0;
        lift.Lift(1) = Lift1;
        lift.A_dt_q(0) = fctPre[fctNo].get<A_dt_q_0>().data();
        lift.A_dt_q(1) = fctPre[fctNo].get<A_dt_q_1>().data();
        lift.B_dt_q(0) = fctPre[fctNo].get<B_dt_q_0>().data();
        lift.B_dt_q(1) = fctPre[fctNo].get<B_dt_q_1>().data();
        lift.n_q = fct[fctNo].get<Normal>().data()->data();
        lift.w = fctRule.weights().data();
        for (int i = 0; i < 2; ++i) {
            lift.E_q(i) = E_q[info.localNo[i]].data();
            lift.L_q(i) = (i == 0) ? L_q0 : L_q1;
            lift.Minv(i) = Minv[i];
        }
        lift.execute(0);
        lift.execute(1);
    } else {
        kernel::lift_ip lift;
        lift.delta = init::delta::Values;
        lift.nl_q = fct[fctNo].get<NormalLength>().data();
        for (int i = 0; i < 2; ++i) {
            lift.E_q(i) = E_q[info.localNo[i]].data();
            lift.L_q(i) = (i == 0) ? L_q0 : L_q1;
            lift.execute(i);
        }
    }
}

} // namespace tndm
