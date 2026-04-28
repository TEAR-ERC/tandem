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
                                 functional_t<1> lam, functional_t<1> mu0, functional_t<1> mu1,
                                 functional_t<1> viscosity, functional_t<1> relaxation_time,
                                 double theta, std::optional<functional_t<1>> rho, DGMethod method)
    : DGCurvilinearCommon<DomainDimension>(std::move(cl), MinQuadOrder()), method_(method),
      space_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>(), ALIGNMENT),
      materialSpace_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>(), ALIGNMENT),
      fun_lam(make_volume_functional(std::move(lam))),
      fun_mu0(make_volume_functional(std::move(mu0))),
      fun_mu1(make_volume_functional(std::move(mu1))),
      fun_viscosity(make_volume_functional(std::move(viscosity))),
      fun_relaxation_time(make_volume_functional(std::move(relaxation_time))), theta_(theta),
      fun_rho(rho ? make_volume_functional(std::move(*rho)) : one_volume_function) {

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
double Viscoelasticity::compute_g_dt(double dt, double tau, int n) const {
    if (dt <= 0.0 || tau <= 0.0) {
        return 0.0;
    }
    const double ratio = dt / tau;
    return -std::expm1(-ratio) / ratio;
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
Relaxation time: element-local CFL bound

Feeds into DGOperator's global MPI reduction to find the minimum τ across
all processes. The result constrains Δt ≤ θ·τ_min for temporal accuracy.

The relaxation-time functional is evaluated at each volume quadrature point.
*/
void Viscoelasticity::local_relaxation_time(std::size_t elNo, double& relaxation_time_global,
                                            LinearAllocator<double>& scratch) {
    base::prepare_volume(elNo, scratch);

    alignas(ALIGNMENT) double tau_raw[tensor::mu1_Q::size()];
    auto tau = Matrix<double>(tau_raw, 1, volRule.size());
    fun_relaxation_time(elNo, tau);

    // Store at all quad points and find element-local minimum
    auto tau_field = volPre[elNo].get<relaxation_time_local>().data();
    for (std::size_t q = 0; q < volRule.size(); ++q) {
        tau_field[q] = tau(0, q);
    }
    relaxation_time_global =
        std::min(relaxation_time_global, *std::min_element(tau_field, tau_field + volRule.size()));
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
