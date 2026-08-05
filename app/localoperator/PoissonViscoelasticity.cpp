#include "PoissonViscoelasticity.h"
#include "config.h"
#include "kernels/poisson_viscoelasticity/init.h"
#include "kernels/poisson_viscoelasticity/kernel.h"
#include "kernels/poisson_viscoelasticity/tensor.h"

#include "basis/WarpAndBlend.h"
#include "form/BC.h"
#include "form/DGCurvilinearCommon.h"
#include "form/InverseInequality.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "tensor/EigenMap.h"
#include "util/LinearAllocator.h"

#include <Eigen/Core>
#include <Eigen/LU>
#include <cassert>
#include <cmath>
#include <limits>

namespace tensor = tndm::poisson_viscoelasticity::tensor;
namespace init = tndm::poisson_viscoelasticity::init;
namespace kernel = tndm::poisson_viscoelasticity::kernel;

namespace tndm {

PoissonViscoelasticity::PoissonViscoelasticity(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                                               functional_t<1> mu0, functional_t<1> mu1,
                                               functional_t<1> eta, double theta, DGMethod method)
    : DGCurvilinearCommon<DomainDimension>(std::move(cl), MinQuadOrder()), method_(method),
      space_(PolynomialDegree, ALIGNMENT),
      materialSpace_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>(), ALIGNMENT),
      fun_mu0(make_volume_functional(std::move(mu0))),
      fun_mu1(make_volume_functional(std::move(mu1))),
      fun_eta(make_volume_functional(std::move(eta))), fun_force(zero_volume_function),
      fun_dirichlet(zero_facet_function), fun_slip(zero_facet_function),
      fun_traction(zero_facet_function), theta_(theta) {

    Minv_ = space_.inverseMassMatrix();
    E_Q = space_.evaluateBasisAt(volRule.points());
    E_Q_T = space_.evaluateBasisAt(volRule.points(), {1, 0});
    Dxi_Q = space_.evaluateGradientAt(volRule.points());

    negative_E_Q_T = Managed<Matrix<double>>(E_Q_T.shape(), std::size_t{ALIGNMENT});
    EigenMap(negative_E_Q_T) = -EigenMap(E_Q_T);

    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto points = cl_->facetParam(f, fctRule.points());
        E_q.emplace_back(space_.evaluateBasisAt(points));
        E_q_T.emplace_back(space_.evaluateBasisAt(points, {1, 0}));
        Dxi_q.emplace_back(space_.evaluateGradientAt(points));
        Dxi_q_120.emplace_back(space_.evaluateGradientAt(points, {1, 2, 0}));
        matE_q_T.emplace_back(materialSpace_.evaluateBasisAt(points, {1, 0}));

        negative_E_q_T.emplace_back(space_.evaluateBasisAt(points, {1, 0}));
        auto E = EigenMap(negative_E_q_T.back());
        E = -E;
    }

    matE_Q_T = materialSpace_.evaluateBasisAt(volRule.points(), {1, 0});
    matDxi_Q = materialSpace_.evaluateGradientAt(volRule.points());
}

// facet history accessors

double* PoissonViscoelasticity::gdt_q(std::size_t fctNo, int side) {
    return side == 1 ? fctHist[fctNo].get<Gdt_q1>().data() : fctHist[fctNo].get<Gdt_q0>().data();
}
double* PoissonViscoelasticity::ratio_q(std::size_t fctNo, int side) {
    return side == 1 ? fctHist[fctNo].get<Ratio_q1>().data()
                     : fctHist[fctNo].get<Ratio_q0>().data();
}
double* PoissonViscoelasticity::mu1_g_q(std::size_t fctNo, int side) {
    return side == 1 ? fctHist[fctNo].get<Mu1G_q1>().data() : fctHist[fctNo].get<Mu1G_q0>().data();
}
double* PoissonViscoelasticity::mu1_r_q(std::size_t fctNo, int side) {
    return side == 1 ? fctHist[fctNo].get<Mu1R_q1>().data() : fctHist[fctNo].get<Mu1R_q0>().data();
}
double* PoissonViscoelasticity::eps_old_q(std::size_t fctNo, int side) {
    return side == 1 ? fctHist[fctNo].get<EpsOld_q1>().data()->data()
                     : fctHist[fctNo].get<EpsOld_q0>().data()->data();
}
double const* PoissonViscoelasticity::eps_old_q(std::size_t fctNo, int side) const {
    return side == 1 ? fctHist[fctNo].get<EpsOld_q1>().data()->data()
                     : fctHist[fctNo].get<EpsOld_q0>().data()->data();
}
double* PoissonViscoelasticity::eps_new_q(std::size_t fctNo, int side) {
    return side == 1 ? fctHist[fctNo].get<EpsNew_q1>().data()->data()
                     : fctHist[fctNo].get<EpsNew_q0>().data()->data();
}
double const* PoissonViscoelasticity::eps_new_q(std::size_t fctNo, int side) const {
    return side == 1 ? fctHist[fctNo].get<EpsNew_q1>().data()->data()
                     : fctHist[fctNo].get<EpsNew_q0>().data()->data();
}
double* PoissonViscoelasticity::q_old_q(std::size_t fctNo, int side) {
    return side == 1 ? fctHist[fctNo].get<QOld_q1>().data()->data()
                     : fctHist[fctNo].get<QOld_q0>().data()->data();
}
double* PoissonViscoelasticity::q_new_q(std::size_t fctNo, int side) {
    return side == 1 ? fctHist[fctNo].get<QNew_q1>().data()->data()
                     : fctHist[fctNo].get<QNew_q0>().data()->data();
}
double const* PoissonViscoelasticity::q_new_q(std::size_t fctNo, int side) const {
    return side == 1 ? fctHist[fctNo].get<QNew_q1>().data()->data()
                     : fctHist[fctNo].get<QNew_q0>().data()->data();
}

// mass matrices (unchanged)

void PoissonViscoelasticity::compute_mass_matrix(std::size_t elNo, double* M) const {
    kernel::massMatrix mm;
    mm.E_Q = E_Q.data();
    mm.J_Q = vol[elNo].get<AbsDetJ>().data();
    mm.M = M;
    mm.W = volRule.weights().data();
    mm.execute();
}

void PoissonViscoelasticity::compute_inverse_mass_matrix(std::size_t elNo, double* Minv) const {
    compute_mass_matrix(elNo, Minv);

    auto J_Q = vol[elNo].get<AbsDetJ>();
    alignas(ALIGNMENT) double Jinv_Q[tensor::Jinv_Q::size()] = {};
    for (unsigned q = 0; q < tensor::Jinv_Q::Shape[0]; ++q) {
        Jinv_Q[q] = 1.0 / J_Q[q];
    }

    kernel::MinvWA wa;
    wa.E_Q = E_Q.data();
    wa.Jinv_Q = Jinv_Q;
    wa.MinvRef = Minv_.data();
    wa.MinvWA = Minv;
    wa.W = volRule.weights().data();
    wa.execute();
}

// mu_eff-weighted facet quantities (K -> mu_eff)

void PoissonViscoelasticity::compute_K_Dx_q(std::size_t fctNo, FacetInfo const& info,
                                            std::array<double*, 2> K_Dx_q) const {
    kernel::K_Dx_q dx;
    for (int i = 0; i < 2; ++i) {
        if (K_Dx_q[i]) {
            auto JInv = (i == 1) ? fct[fctNo].get<JInv1>() : fct[fctNo].get<JInv0>();
            dx.G_q = JInv.data()->data();
            dx.matE_q_T = matE_q_T[info.localNo[i]].data();
            dx.mu_eff = material[info.up[i]].get<MuEff>().data();
            dx.K_Dx_q(0) = K_Dx_q[i];
            dx.Dxi_q(0) = Dxi_q[info.localNo[i]].data();
            dx.execute();
        }
    }
}

void PoissonViscoelasticity::compute_K_q(std::size_t fctNo, FacetInfo const& info,
                                         std::array<double*, 2> K_q) const {
    kernel::K_q kw;
    for (int i = 0; i < 2; ++i) {
        if (K_q[i]) {
            kw.matE_q_T = matE_q_T[info.localNo[i]].data();
            kw.mu_eff = material[info.up[i]].get<MuEff>().data();
            kw.K_q(0) = K_q[i];
            kw.execute();
        }
    }
}

// preparation

void PoissonViscoelasticity::begin_preparation(std::size_t numElements,
                                               std::size_t numLocalElements,
                                               std::size_t numLocalFacets) {
    base::begin_preparation(numElements, numLocalElements, numLocalFacets);

    numElements_ = numElements;
    numLocalElements_ = numLocalElements;
    numLocalFacets_ = numLocalFacets;

    auto const matNbf = materialSpace_.numBasisFunctions();

    material.setStorage(std::make_shared<material_vol_t>(numElements * matNbf), 0u, numElements,
                        matNbf);

    volPre.setStorage(std::make_shared<vol_pre_t>(numElements * volRule.size()), 0u, numElements,
                      volRule.size());

    volHist.setStorage(std::make_shared<vol_hist_t>(numElements * volRule.size()), 0u, numElements,
                       volRule.size());

    displacement.setStorage(std::make_shared<displ_t>(numElements * tensor::U_old::size()), 0u,
                            numElements, tensor::U_old::size());

    fctPre.setStorage(std::make_shared<fct_pre_t>(numLocalFacets * fctRule.size()), 0u,
                      numLocalFacets, fctRule.size());

    fctHist.setStorage(std::make_shared<fct_hist_t>(numLocalFacets * fctRule.size()), 0u,
                       numLocalFacets, fctRule.size());

    fctCache_.assign(numLocalFacets, CachedFacet{});
    penalty_.resize(numLocalFacets);
}

void PoissonViscoelasticity::prepare_volume(std::size_t elNo, LinearAllocator<double>& scratch) {
    base::prepare_volume(elNo, scratch);

    auto const matNbf = materialSpace_.numBasisFunctions();

    alignas(ALIGNMENT) double Mmem[tensor::matM::size()];
    kernel::project_K_lhs krnl_lhs;
    krnl_lhs.matE_Q_T = matE_Q_T.data();
    krnl_lhs.J_Q = vol[elNo].get<AbsDetJ>().data();
    krnl_lhs.matM = Mmem;
    krnl_lhs.W = volRule.weights().data();
    krnl_lhs.execute();

    using MMap = Eigen::Map<Eigen::Matrix<double, tensor::matM::Shape[0], tensor::matM::Shape[1]>,
                            Eigen::Unaligned,
                            Eigen::OuterStride<init::matM::Stop[0] - init::matM::Start[0]>>;
    using KMap = Eigen::Map<Eigen::Matrix<double, tensor::K::Shape[0], 1>, Eigen::Unaligned,
                            Eigen::InnerStride<1>>;
    auto lu = MMap(Mmem).fullPivLu();

    // Project mu_0, mu_1 and eta with the same factorization.
    alignas(ALIGNMENT) double coeff_Q_raw[tensor::K_Q::size()];
    auto coeff_Q = Matrix<double>(coeff_Q_raw, 1, volRule.size());

    auto const project = [&](volume_functional_t const& fun, double* target) {
        fun(elNo, coeff_Q);
        kernel::project_K_rhs krnl_rhs;
        krnl_rhs.matE_Q_T = matE_Q_T.data();
        krnl_rhs.J_Q = vol[elNo].get<AbsDetJ>().data();
        krnl_rhs.K = target;
        krnl_rhs.K_Q = coeff_Q_raw;
        krnl_rhs.W = volRule.weights().data();
        krnl_rhs.execute();
        auto v = KMap(target);
        v = lu.solve(v);
    };

    auto mu0 = material[elNo].get<K>().data();
    auto mu1 = material[elNo].get<Mu1>().data();
    auto tau = material[elNo].get<Tau>().data();

    project(fun_mu0, mu0);
    project(fun_mu1, mu1);
    project(fun_eta, tau); // holds eta for now

    // tau = eta / mu_1.  mu_1 == 0 marks a purely elastic node: tau = infinity.
    for (std::size_t i = 0; i < matNbf; ++i) {
        if (mu1[i] > 0.0) {
            tau[i] = tau[i] / mu1[i];
        } else {
            mu1[i] = 0.0;
            tau[i] = std::numeric_limits<double>::infinity();
        }
    }

    compute_time_coefficients(elNo);
    compute_history_weights_Q(elNo);
}

void PoissonViscoelasticity::prepare_skeleton(std::size_t fctNo, FacetInfo const& info,
                                              LinearAllocator<double>& scratch) {
    base::prepare_skeleton(fctNo, info, scratch);

    fctCache_[fctNo].localNo = {info.localNo[0], info.localNo[1]};
    fctCache_[fctNo].up = {info.up[0], info.up[1]};
    fctCache_[fctNo].skeleton = true;

    for (int side = 0; side < 2; ++side) {
        compute_K_G_q(fctNo, side, info.up[side], info.localNo[side]);
        compute_history_weights_q(fctNo, side, info.up[side], info.localNo[side]);
    }
}

void PoissonViscoelasticity::prepare_boundary(std::size_t fctNo, FacetInfo const& info,
                                              LinearAllocator<double>& scratch) {
    base::prepare_boundary(fctNo, info, scratch);

    fctCache_[fctNo].localNo = {info.localNo[0], info.localNo[0]};
    fctCache_[fctNo].up = {info.up[0], info.up[0]};
    fctCache_[fctNo].skeleton = false;

    compute_K_G_q(fctNo, 0, info.up[0], info.localNo[0]);
    compute_history_weights_q(fctNo, 0, info.up[0], info.localNo[0]);
}

void PoissonViscoelasticity::prepare_volume_post_skeleton(std::size_t elNo,
                                                          LinearAllocator<double>& scratch) {
    base::prepare_volume_post_skeleton(elNo, scratch);
    compute_J_W_K_Q(elNo);
}

void PoissonViscoelasticity::prepare_penalty(std::size_t fctNo, FacetInfo const& info,
                                             LinearAllocator<double>&) {
    fctCache_[fctNo].localNo = {info.localNo[0], info.localNo[info.up[0] != info.up[1] ? 1 : 0]};
    fctCache_[fctNo].up = {info.up[0], info.up[1]};
    fctCache_[fctNo].skeleton = info.up[0] != info.up[1];
    compute_penalty(fctNo);
}

// time-dependent coefficients

void PoissonViscoelasticity::compute_time_coefficients(std::size_t elNo) {
    auto const matNbf = materialSpace_.numBasisFunctions();
    auto tau = material[elNo].get<Tau>().data();
    auto gdt = material[elNo].get<Gdt>().data();
    auto ratio = material[elNo].get<Ratio>().data();

    for (std::size_t i = 0; i < matNbf; ++i) {
        double const t = tau[i];
        if (!std::isfinite(t) || t <= 0.0 || dt_ <= 0.0) {
            // Purely elastic node, or dt not set yet: ratio = g = 1 gives
            // mu_eff = mu_0 + mu_1 (unrelaxed) and sigmaHat = 2 mu_1 (q - eps),
            // which vanishes because q tracks eps exactly in that limit.
            ratio[i] = 1.0;
            gdt[i] = 1.0;
            continue;
        }
        double const x = dt_ / t;
        ratio[i] = std::exp(-x);
        // g = (1 - exp(-x)) / x, evaluated stably for small x.
        gdt[i] = (x < 1.0e-8) ? (1.0 - 0.5 * x + x * x / 6.0) : (-std::expm1(-x) / x);
    }

    kernel::get_mu_eff k;
    k.mu_eff = material[elNo].get<MuEff>().data();
    k.K = material[elNo].get<K>().data();
    k.mu1 = material[elNo].get<Mu1>().data();
    k.g_dt = gdt;
    k.execute();
}

void PoissonViscoelasticity::compute_history_weights_Q(std::size_t elNo) {
    kernel::history_weights_Q k;
    k.g_dt = material[elNo].get<Gdt>().data();
    k.ratio = material[elNo].get<Ratio>().data();
    k.mu1 = material[elNo].get<Mu1>().data();
    k.matE_Q_T = matE_Q_T.data();
    k.J_Q = vol[elNo].get<AbsDetJ>().data();
    k.W = volRule.weights().data();
    k.gdt_Q = volHist[elNo].get<GdtQ>().data();
    k.ratio_Q = volHist[elNo].get<RatioQ>().data();
    k.mu1_g_W_J_Q = volHist[elNo].get<Mu1GWJ_Q>().data();
    k.mu1_r_W_J_Q = volHist[elNo].get<Mu1RWJ_Q>().data();
    k.execute();
}

void PoissonViscoelasticity::compute_history_weights_q(std::size_t fctNo, int side,
                                                       std::size_t elNo, std::size_t localNo) {
    kernel::history_weights_q k;
    k.g_dt = material[elNo].get<Gdt>().data();
    k.ratio = material[elNo].get<Ratio>().data();
    k.mu1 = material[elNo].get<Mu1>().data();
    k.matE_q_T = matE_q_T[localNo].data();
    k.gdt_q(0) = gdt_q(fctNo, side);
    k.ratio_q(0) = ratio_q(fctNo, side);
    k.mu1_g_q(0) = mu1_g_q(fctNo, side);
    k.mu1_r_q(0) = mu1_r_q(fctNo, side);
    k.execute();
}

void PoissonViscoelasticity::compute_J_W_K_Q(std::size_t elNo) {
    kernel::J_W_K_Q krnl;
    krnl.J_W_K_Q = volPre[elNo].get<AbsDetJWK>().data()->data();
    krnl.J_Q = vol[elNo].get<AbsDetJ>().data();
    krnl.mu_eff = material[elNo].get<MuEff>().data();
    krnl.matE_Q_T = matE_Q_T.data();
    krnl.W = volRule.weights().data();
    krnl.execute();
}

void PoissonViscoelasticity::compute_K_G_q(std::size_t fctNo, int side, std::size_t elNo,
                                           std::size_t localNo) {
    kernel::K_G_q k;
    k.G_q =
        side == 1 ? fct[fctNo].get<JInv1>().data()->data() : fct[fctNo].get<JInv0>().data()->data();
    k.mu_eff = material[elNo].get<MuEff>().data();
    k.K_G_q(0) = side == 1 ? fctPre[fctNo].get<KJInv1>().data()->data()
                           : fctPre[fctNo].get<KJInv0>().data()->data();
    k.matE_q_T = matE_q_T[localNo].data();
    k.execute();
}

void PoissonViscoelasticity::compute_penalty(std::size_t fctNo) {
    auto const& c = fctCache_[fctNo];
    auto const p = [&](int side) {
        auto field = material[c.up[side]].get<MuEff>().data();
        auto k0 = *std::min_element(field, field + materialSpace_.numBasisFunctions());
        auto k1 = *std::max_element(field, field + materialSpace_.numBasisFunctions());
        constexpr double c_N_1 = InverseInequality<Dim>::trace_constant(PolynomialDegree - 1);
        return (Dim + 1) * c_N_1 * (area_[fctNo] / volume_[c.up[side]]) * (k1 * k1 / k0);
    };

    if (c.skeleton) {
        penalty_[fctNo] = (p(0) + p(1)) / 4.0;
    } else {
        penalty_[fctNo] = p(0);
    }
}

void PoissonViscoelasticity::update_time_dependent_precomputation_volume(std::size_t elNo) {
    compute_time_coefficients(elNo);
    compute_history_weights_Q(elNo);
    compute_J_W_K_Q(elNo);
}

void PoissonViscoelasticity::update_time_dependent_precomputation_skeleton(std::size_t fctNo) {
    auto const& c = fctCache_[fctNo];
    for (int side = 0; side < 2; ++side) {
        compute_K_G_q(fctNo, side, c.up[side], c.localNo[side]);
        compute_history_weights_q(fctNo, side, c.up[side], c.localNo[side]);
    }
    // The penalty depends on mu_eff and therefore on dt.
    compute_penalty(fctNo);
}

void PoissonViscoelasticity::update_time_dependent_precomputation_boundary(std::size_t fctNo) {
    auto const& c = fctCache_[fctNo];
    compute_K_G_q(fctNo, 0, c.up[0], c.localNo[0]);
    compute_history_weights_q(fctNo, 0, c.up[0], c.localNo[0]);
    compute_penalty(fctNo);
}

void PoissonViscoelasticity::local_relaxation_time(std::size_t elNo, double& dt_min,
                                                   LinearAllocator<double>& scratch) {
    // Called before prepare_volume, so the material is not projected yet. The
    // volume functionals still need mapped quadrature-point coordinates
    // (vol[elNo].get<Coords>()), which are only populated by prepare_volume --
    // so run it here first, exactly like Viscoelasticity::local_relaxation_time.
    base::prepare_volume(elNo, scratch);

    alignas(ALIGNMENT) double mu1_Q[tensor::K_Q::size()];
    alignas(ALIGNMENT) double eta_Q[tensor::K_Q::size()];
    auto mu1_mat = Matrix<double>(mu1_Q, 1, volRule.size());
    auto eta_mat = Matrix<double>(eta_Q, 1, volRule.size());
    fun_mu1(elNo, mu1_mat);
    fun_eta(elNo, eta_mat);

    for (std::size_t q = 0; q < volRule.size(); ++q) {
        if (mu1_Q[q] > 0.0 && eta_Q[q] > 0.0) {
            double const tau = eta_Q[q] / mu1_Q[q];
            if (tau < dt_min) {
                dt_min = tau;
            }
        }
    }
}

// viscoelastic state

void PoissonViscoelasticity::initialize_displacement_field(std::size_t elNo) {
    auto u = displacement[elNo].get<Displacement>().data();
    for (std::size_t i = 0; i < tensor::U_old::size(); ++i) {
        u[i] = 0.0;
    }
}

void PoissonViscoelasticity::store_displacement_field(std::size_t elNo, double const* u) {
    auto target = displacement[elNo].get<Displacement>().data();
    for (std::size_t i = 0; i < block_size(); ++i) {
        target[i] = u[i];
    }
}

void PoissonViscoelasticity::initialize_strain_tensor_Q(std::size_t elNo) {
    auto const n = Dim * volRule.size();
    auto zero = [n](double* p) {
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = 0.0;
        }
    };
    zero(volHist[elNo].get<EpsOldQ>().data()->data());
    zero(volHist[elNo].get<EpsNewQ>().data()->data());
    zero(volHist[elNo].get<QOldQ>().data()->data());
    zero(volHist[elNo].get<QNewQ>().data()->data());
}

void PoissonViscoelasticity::initialize_strain_tensor_q(std::size_t fctNo) {
    auto const n = Dim * fctRule.size();
    auto zero = [n](double* p) {
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = 0.0;
        }
    };
    for (int side = 0; side < 2; ++side) {
        zero(eps_old_q(fctNo, side));
        zero(eps_new_q(fctNo, side));
        zero(q_old_q(fctNo, side));
        zero(q_new_q(fctNo, side));
    }
}

void PoissonViscoelasticity::initialize_partial_strain_Q(std::size_t elNo) {
    // q(0) = eps(0): instantaneous unrelaxed response at t = 0.
    auto const n = Dim * volRule.size();
    auto eps = volHist[elNo].get<EpsNewQ>().data()->data();
    auto q_new = volHist[elNo].get<QNewQ>().data()->data();
    auto q_old = volHist[elNo].get<QOldQ>().data()->data();
    auto eps_old = volHist[elNo].get<EpsOldQ>().data()->data();
    for (std::size_t i = 0; i < n; ++i) {
        q_new[i] = eps[i];
        q_old[i] = eps[i];
        eps_old[i] = eps[i];
    }
}

void PoissonViscoelasticity::initialize_partial_strain_q(std::size_t fctNo) {
    auto const n = Dim * fctRule.size();
    for (int side = 0; side < 2; ++side) {
        auto eps = eps_new_q(fctNo, side);
        auto qn = q_new_q(fctNo, side);
        auto qo = q_old_q(fctNo, side);
        auto eo = eps_old_q(fctNo, side);
        for (std::size_t i = 0; i < n; ++i) {
            qn[i] = eps[i];
            qo[i] = eps[i];
            eo[i] = eps[i];
        }
    }
}

void PoissonViscoelasticity::compute_deviatoric_strain_Q(std::size_t elNo) {
    alignas(ALIGNMENT) double Dx_Q[tensor::Dx_Q::size()];
    kernel::Dx_Q dx;
    dx.Dx_Q = Dx_Q;
    dx.Dxi_Q = Dxi_Q.data();
    dx.G_Q = vol[elNo].get<JInv>().data()->data();
    dx.execute();

    kernel::strain_Q k;
    k.Dx_Q = Dx_Q;
    k.U_old = displacement[elNo].get<Displacement>().data();
    k.eps_n_Q = volHist[elNo].get<EpsNewQ>().data()->data();
    k.execute();
}

void PoissonViscoelasticity::compute_deviatoric_strain_q(std::size_t fctNo, FacetInfo const& info) {
    int const nsides = (info.up[0] != info.up[1]) ? 2 : 1;
    for (int side = 0; side < nsides; ++side) {
        kernel::strain_q k;
        k.G_q = side == 1 ? fct[fctNo].get<JInv1>().data()->data()
                          : fct[fctNo].get<JInv0>().data()->data();
        k.Dxi_q(0) = Dxi_q[info.localNo[side]].data();
        k.U_old = displacement[info.up[side]].get<Displacement>().data();
        k.eps_n_q(0) = eps_new_q(fctNo, side);
        k.execute();
    }
}

void PoissonViscoelasticity::compute_partial_strain_Q(std::size_t elNo) {
    kernel::update_history_Q k;
    k.ratio_Q = volHist[elNo].get<RatioQ>().data();
    k.gdt_Q = volHist[elNo].get<GdtQ>().data();
    k.q_n_Q = volHist[elNo].get<QOldQ>().data()->data();
    k.eps_n_Q = volHist[elNo].get<EpsOldQ>().data()->data();
    k.eps_new_Q = volHist[elNo].get<EpsNewQ>().data()->data();
    k.q_new_Q = volHist[elNo].get<QNewQ>().data()->data();
    k.execute();
}

void PoissonViscoelasticity::compute_partial_strain_q(std::size_t fctNo, FacetInfo const& info) {
    int const nsides = (info.up[0] != info.up[1]) ? 2 : 1;
    for (int side = 0; side < nsides; ++side) {
        kernel::update_history_q k;
        k.ratio_q(0) = ratio_q(fctNo, side);
        k.gdt_q(0) = gdt_q(fctNo, side);
        k.q_n_q(0) = q_old_q(fctNo, side);
        k.eps_n_q(0) = eps_old_q(fctNo, side);
        k.eps_new_q(0) = eps_new_q(fctNo, side);
        k.q_new_q(0) = q_new_q(fctNo, side);
        k.execute();
    }
}

void PoissonViscoelasticity::update_deviatoric_strain_Q(std::size_t elNo) {
    auto const n = Dim * volRule.size();
    auto src = volHist[elNo].get<EpsNewQ>().data()->data();
    auto dst = volHist[elNo].get<EpsOldQ>().data()->data();
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

void PoissonViscoelasticity::update_deviatoric_strain_q(std::size_t fctNo) {
    auto const n = Dim * fctRule.size();
    for (int side = 0; side < 2; ++side) {
        auto src = eps_new_q(fctNo, side);
        auto dst = eps_old_q(fctNo, side);
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = src[i];
        }
    }
}

void PoissonViscoelasticity::update_partial_strain_Q(std::size_t elNo) {
    auto const n = Dim * volRule.size();
    auto src = volHist[elNo].get<QNewQ>().data()->data();
    auto dst = volHist[elNo].get<QOldQ>().data()->data();
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

void PoissonViscoelasticity::update_partial_strain_q(std::size_t fctNo) {
    auto const n = Dim * fctRule.size();
    for (int side = 0; side < 2; ++side) {
        auto src = q_new_q(fctNo, side);
        auto dst = q_old_q(fctNo, side);
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = src[i];
        }
    }
}

void PoissonViscoelasticity::compute_sigma_hist_q(std::size_t fctNo, int side,
                                                  double* sigma_hist) const {
    kernel::sigma_hist_q k;
    k.mu1_r_q(0) =
        side == 1 ? fctHist[fctNo].get<Mu1R_q1>().data() : fctHist[fctNo].get<Mu1R_q0>().data();
    k.mu1_g_q(0) =
        side == 1 ? fctHist[fctNo].get<Mu1G_q1>().data() : fctHist[fctNo].get<Mu1G_q0>().data();
    k.q_n_q(0) = q_new_q(fctNo, side);
    k.eps_n_q(0) = eps_new_q(fctNo, side);
    k.sigma_hist_q(0) = sigma_hist;
    k.execute();
}

// assembly

bool PoissonViscoelasticity::assemble_volume(std::size_t elNo, Matrix<double>& A00,
                                             LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double Dx_Q[tensor::Dx_Q::size()];

    assert(volRule.size() == tensor::W::Shape[0]);
    assert(Dxi_Q.shape(0) == tensor::Dxi_Q::Shape[0]);
    assert(Dxi_Q.shape(1) == tensor::Dxi_Q::Shape[1]);
    assert(Dxi_Q.shape(2) == tensor::Dxi_Q::Shape[2]);

    kernel::Dx_Q dx;
    dx.Dx_Q = Dx_Q;
    dx.Dxi_Q = Dxi_Q.data();
    dx.G_Q = vol[elNo].get<JInv>().data()->data();
    dx.execute();

    kernel::assembleVolume krnl;
    krnl.A = A00.data();
    krnl.Dx_Q = Dx_Q;
    krnl.mu_eff = material[elNo].get<MuEff>().data();
    krnl.matE_Q_T = matE_Q_T.data();
    krnl.J_Q = vol[elNo].get<AbsDetJ>().data();
    krnl.W = volRule.weights().data();
    krnl.execute();
    return true;
}

bool PoissonViscoelasticity::assemble_skeleton(std::size_t fctNo, FacetInfo const& info,
                                               Matrix<double>& A00, Matrix<double>& A01,
                                               Matrix<double>& A10, Matrix<double>& A11,
                                               LinearAllocator<double>& scratch) const {
    assert(fctRule.size() == tensor::w::Shape[0]);

    alignas(ALIGNMENT) double K_Dx_q0[tensor::K_Dx_q::size(0)];
    alignas(ALIGNMENT) double K_Dx_q1[tensor::K_Dx_q::size(1)];
    auto K_Dx_q = std::array<double*, 2>{K_Dx_q0, K_Dx_q1};
    compute_K_Dx_q(fctNo, info, K_Dx_q);

    alignas(ALIGNMENT) double L_q[2][std::max(tensor::L_q::size(0), tensor::L_q::size(1))];

    if (method_ == DGMethod::BR2) {
        alignas(ALIGNMENT) double Lift0[tensor::Lift::size(0)];
        alignas(ALIGNMENT) double Lift1[tensor::Lift::size(1)];
        alignas(ALIGNMENT) double Minv[2][tensor::M::size()];
        for (int i = 0; i < 2; ++i) {
            compute_inverse_mass_matrix(info.up[i], Minv[i]);
        }

        alignas(ALIGNMENT) double K_q0[tensor::K_q::size(0)];
        alignas(ALIGNMENT) double K_q1[tensor::K_q::size(1)];
        auto K_q = std::array<double*, 2>{K_q0, K_q1};
        compute_K_q(fctNo, info, K_q);

        kernel::lift_skeleton lift;
        lift.Lift(0) = Lift0;
        lift.Lift(1) = Lift1;
        lift.n_q = fct[fctNo].get<Normal>().data()->data();
        lift.w = fctRule.weights().data();
        for (int i = 0; i < 2; ++i) {
            lift.K_q(i) = K_q[i];
            lift.L_q(i) = L_q[i];
            lift.Minv(i) = Minv[i];
            lift.E_q(i) = E_q[info.localNo[i]].data();
        }
        lift.execute(0);
        lift.execute(1);
    } else { // IP
        kernel::lift_ip lift;
        lift.nl_q = fct[fctNo].get<NormalLength>().data();
        for (int i = 0; i < 2; ++i) {
            lift.L_q(i) = L_q[i];
            lift.E_q(i) = E_q[info.localNo[i]].data();
        }
        lift.execute(0);
        lift.execute(1);
    }

    kernel::assembleSurface assemble;
    assemble.c00 = -0.5;
    assemble.c01 = -assemble.c00;
    assemble.c10 = epsilon * 0.5;
    assemble.c11 = -assemble.c10;
    assemble.c20 = penalty(fctNo);
    assemble.c21 = -assemble.c20;
    assemble.a(0, 0) = A00.data();
    assemble.a(0, 1) = A01.data();
    assemble.a(1, 0) = A10.data();
    assemble.a(1, 1) = A11.data();
    for (int i = 0; i < 2; ++i) {
        assemble.K_Dx_q(i) = K_Dx_q[i];
        assemble.E_q(i) = E_q[info.localNo[i]].data();
        assemble.L_q(i) = L_q[i];
    }
    assemble.n_q = fct[fctNo].get<Normal>().data()->data();
    assemble.w = fctRule.weights().data();
    assemble.execute(0, 0);
    assemble.execute(0, 1);
    assemble.execute(1, 0);
    assemble.execute(1, 1);

    return true;
}

bool PoissonViscoelasticity::assemble_boundary(std::size_t fctNo, FacetInfo const& info,
                                               Matrix<double>& A00,
                                               LinearAllocator<double>& scratch) const {
    if (info.bc == BC::Natural) {
        return false;
    }

    assert(fctRule.size() == tensor::w::Shape[0]);

    alignas(ALIGNMENT) double L0[tensor::L_q::size(0)];
    if (method_ == DGMethod::BR2) {
        alignas(ALIGNMENT) double Lift0[tensor::Lift::size(0)];
        alignas(ALIGNMENT) double Minv0[tensor::M::size()];
        compute_inverse_mass_matrix(info.up[0], Minv0);

        alignas(ALIGNMENT) double K_q[tensor::K_q::size(0)];
        compute_K_q(fctNo, info, {K_q, nullptr});

        kernel::lift_boundary lift;
        lift.Lift(0) = Lift0;
        lift.K_q(0) = K_q;
        lift.L_q(0) = L0;
        lift.Minv(0) = Minv0;
        lift.E_q(0) = E_q[info.localNo[0]].data();
        lift.n_q = fct[fctNo].get<Normal>().data()->data();
        lift.w = fctRule.weights().data();
        lift.execute();
    } else { // IP
        kernel::lift_ip lift;
        lift.nl_q = fct[fctNo].get<NormalLength>().data();
        lift.L_q(0) = L0;
        lift.E_q(0) = E_q[info.localNo[0]].data();
        lift.execute(0);
    }

    alignas(ALIGNMENT) double K_Dx_q0[tensor::K_Dx_q::size(0)];
    compute_K_Dx_q(fctNo, info, {K_Dx_q0, nullptr});

    kernel::assembleSurface assemble;
    assemble.c00 = -1.0;
    assemble.c10 = epsilon;
    assemble.c20 = penalty(fctNo);
    assemble.a(0, 0) = A00.data();
    assemble.K_Dx_q(0) = K_Dx_q0;
    assemble.E_q(0) = E_q[info.localNo[0]].data();
    assemble.L_q(0) = L0;
    assemble.n_q = fct[fctNo].get<Normal>().data()->data();
    assemble.w = fctRule.weights().data();
    assemble.execute(0, 0);
    return true;
}

// right-hand side

bool PoissonViscoelasticity::rhs_volume(std::size_t elNo, Vector<double>& B,
                                        LinearAllocator<double>& scratch) const {
    assert(tensor::b::Shape[0] == tensor::A::Shape[0]);

    alignas(ALIGNMENT) double F_Q_raw[tensor::F_Q::size()];
    assert(tensor::F_Q::size() == volRule.size());
    auto F_Q = Matrix<double>(F_Q_raw, 1, tensor::F_Q::Shape[0]);
    fun_force(elNo, F_Q);

    kernel::rhsVolume rhs;
    rhs.E_Q = E_Q.data();
    rhs.F_Q = F_Q_raw;
    rhs.J_Q = vol[elNo].get<AbsDetJ>().data();
    rhs.W = volRule.weights().data();
    rhs.b = B.data();
    rhs.execute();
    return true;
}

bool PoissonViscoelasticity::rhs_history_volume(std::size_t elNo, Vector<double>& B,
                                                LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double Dx_Q[tensor::Dx_Q::size()];
    kernel::Dx_Q dx;
    dx.Dx_Q = Dx_Q;
    dx.Dxi_Q = Dxi_Q.data();
    dx.G_Q = vol[elNo].get<JInv>().data()->data();
    dx.execute();

    kernel::rhsVolumeHistory rhs;
    rhs.b = B.data();
    rhs.Dx_Q = Dx_Q;
    rhs.mu1_g_W_J_Q = volHist[elNo].get<Mu1GWJ_Q>().data();
    rhs.mu1_r_W_J_Q = volHist[elNo].get<Mu1RWJ_Q>().data();
    rhs.eps_n_Q = volHist[elNo].get<EpsNewQ>().data()->data();
    rhs.q_n_Q = volHist[elNo].get<QNewQ>().data()->data();
    rhs.execute();
    return true;
}

bool PoissonViscoelasticity::bc_skeleton(std::size_t fctNo, BC bc, double f_q_raw[]) const {
    assert(tensor::f_q::size() == fctRule.size());
    auto f_q = Matrix<double>(f_q_raw, 1, tensor::f_q::Shape[0]);
    if (bc == BC::Fault) {
        fun_slip(fctNo, f_q, false);
    } else if (bc == BC::Dirichlet) {
        fun_dirichlet(fctNo, f_q, false);
    } else {
        return false;
    }
    return true;
}

bool PoissonViscoelasticity::bc_boundary(std::size_t fctNo, BC bc, double f_q_raw[]) const {
    assert(tensor::f_q::size() == fctRule.size());
    auto f_q = Matrix<double>(f_q_raw, 1, tensor::f_q::Shape[0]);
    if (bc == BC::Fault) {
        fun_slip(fctNo, f_q, true);
        for (std::size_t q = 0; q < tensor::f_q::Shape[0]; ++q) {
            f_q(0, q) *= 0.5;
        }
    } else if (bc == BC::Dirichlet) {
        fun_dirichlet(fctNo, f_q, true);
    } else {
        return false;
    }
    return true;
}

bool PoissonViscoelasticity::rhs_skeleton(std::size_t fctNo, FacetInfo const& info,
                                          Vector<double>& B0, Vector<double>& B1,
                                          LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    if (!bc_skeleton(fctNo, info.bc, f_q_raw)) {
        return false;
    }

    alignas(ALIGNMENT) double f_lifted_q[tensor::f_lifted_q::size()];
    if (method_ == DGMethod::BR2) {
        alignas(ALIGNMENT) double f_lifted0[tensor::f_lifted::size(0)];
        alignas(ALIGNMENT) double f_lifted1[tensor::f_lifted::size(1)];
        alignas(ALIGNMENT) double Minv[2][tensor::M::size()];
        compute_inverse_mass_matrix(info.up[0], Minv[0]);
        compute_inverse_mass_matrix(info.up[1], Minv[1]);

        alignas(ALIGNMENT) double K_q0[tensor::K_q::size(0)];
        alignas(ALIGNMENT) double K_q1[tensor::K_q::size(1)];
        auto K_q = std::array<double*, 2>{K_q0, K_q1};
        compute_K_q(fctNo, info, K_q);

        kernel::rhs_lift_skeleton lift;
        for (int i = 0; i < 2; ++i) {
            lift.E_q(i) = E_q[info.localNo[i]].data();
            lift.K_q(i) = K_q[i];
            lift.Minv(i) = Minv[i];
        }
        lift.n_q = fct[fctNo].get<Normal>().data()->data();
        lift.f_q = f_q_raw;
        lift.f_lifted(0) = f_lifted0;
        lift.f_lifted(1) = f_lifted1;
        lift.f_lifted_q = f_lifted_q;
        lift.w = fctRule.weights().data();
        lift.execute();
    } else { // IP
        kernel::rhs_lift_ip lift;
        lift.nl_q = fct[fctNo].get<NormalLength>().data();
        lift.f_q = f_q_raw;
        lift.f_lifted_q = f_lifted_q;
        lift.execute();
    }

    alignas(ALIGNMENT) double K_Dx_q0[tensor::K_Dx_q::size(0)];
    alignas(ALIGNMENT) double K_Dx_q1[tensor::K_Dx_q::size(1)];
    compute_K_Dx_q(fctNo, info, {K_Dx_q0, K_Dx_q1});

    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = 0.5 * epsilon;
    rhs.c20 = penalty(fctNo);
    rhs.f_q = f_q_raw;
    rhs.f_lifted_q = f_lifted_q;
    rhs.n_q = fct[fctNo].get<Normal>().data()->data();
    rhs.w = fctRule.weights().data();
    rhs.K_Dx_q(0) = K_Dx_q0;
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.execute();

    rhs.b = B1.data();
    rhs.c20 *= -1.0;
    rhs.K_Dx_q(0) = K_Dx_q1;
    rhs.E_q(0) = E_q[info.localNo[1]].data();
    rhs.execute();

    return true;
}

bool PoissonViscoelasticity::rhs_history_skeleton(std::size_t fctNo, FacetInfo const& info,
                                                  Vector<double>& B0, Vector<double>& B1,
                                                  LinearAllocator<double>& scratch) const {
    // + int_F [[v]] {sigmaHat^n . n},  [[v]] = v_0 - v_1 with n outward from
    // element 0.  Applies on EVERY skeleton facet, not just Dirichlet/fault.
    alignas(ALIGNMENT) double sigma0[tensor::sigma_hist_q::size(0)];
    alignas(ALIGNMENT) double sigma1[tensor::sigma_hist_q::size(1)];
    compute_sigma_hist_q(fctNo, 0, sigma0);
    compute_sigma_hist_q(fctNo, 1, sigma1);

    alignas(ALIGNMENT) double traction_hist_q[tensor::traction_hist_q::size()];
    kernel::tractionHistoryAverageSkeleton avg;
    avg.n_q = fct[fctNo].get<Normal>().data()->data();
    avg.sigma_hist_q(0) = sigma0;
    avg.sigma_hist_q(1) = sigma1;
    avg.traction_hist_q = traction_hist_q;
    avg.execute();

    kernel::rhsFacetHistory rhs;
    rhs.traction_hist_q = traction_hist_q;
    rhs.w = fctRule.weights().data();

    rhs.b = B0.data();
    rhs.c3 = 1.0;
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.execute();

    rhs.b = B1.data();
    rhs.c3 = -1.0;
    rhs.E_q(0) = E_q[info.localNo[1]].data();
    rhs.execute();

    return true;
}

bool PoissonViscoelasticity::rhs_boundary(std::size_t fctNo, FacetInfo const& info,
                                          Vector<double>& B0,
                                          LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    if (!bc_boundary(fctNo, info.bc, f_q_raw)) {
        return false;
    }

    alignas(ALIGNMENT) double f_lifted_q[tensor::f_lifted_q::size()];
    if (method_ == DGMethod::BR2) {
        alignas(ALIGNMENT) double f_lifted0[tensor::f_lifted::size(0)];
        alignas(ALIGNMENT) double M0[tensor::M::size()];
        compute_inverse_mass_matrix(info.up[0], M0);

        alignas(ALIGNMENT) double K_q[tensor::K_q::size(0)];
        compute_K_q(fctNo, info, {K_q, nullptr});

        kernel::rhs_lift_boundary lift;
        lift.E_q(0) = E_q[info.localNo[0]].data();
        lift.n_q = fct[fctNo].get<Normal>().data()->data();
        lift.K_q(0) = K_q;
        lift.Minv(0) = M0;
        lift.f_q = f_q_raw;
        lift.f_lifted(0) = f_lifted0;
        lift.f_lifted_q = f_lifted_q;
        lift.w = fctRule.weights().data();
        lift.execute();
    } else { // IP
        kernel::rhs_lift_ip lift;
        lift.nl_q = fct[fctNo].get<NormalLength>().data();
        lift.f_q = f_q_raw;
        lift.f_lifted_q = f_lifted_q;
        lift.execute();
    }

    alignas(ALIGNMENT) double K_Dx_q0[tensor::K_Dx_q::size(0)];
    compute_K_Dx_q(fctNo, info, {K_Dx_q0, nullptr});

    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = epsilon;
    rhs.c20 = penalty(fctNo);
    rhs.f_q = f_q_raw;
    rhs.f_lifted_q = f_lifted_q;
    rhs.n_q = fct[fctNo].get<Normal>().data()->data();
    rhs.w = fctRule.weights().data();
    rhs.K_Dx_q(0) = K_Dx_q0;
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.execute();
    return true;
}

bool PoissonViscoelasticity::rhs_history_boundary(std::size_t fctNo, FacetInfo const& info,
                                                  Vector<double>& B0,
                                                  LinearAllocator<double>& scratch) const {
    // On a Natural facet the prescribed traction is the TOTAL traction, which
    // already contains the history contribution -- no consistency term here.
    if (info.bc == BC::Natural) {
        return false;
    }

    alignas(ALIGNMENT) double sigma0[tensor::sigma_hist_q::size(0)];
    compute_sigma_hist_q(fctNo, 0, sigma0);

    alignas(ALIGNMENT) double traction_hist_q[tensor::traction_hist_q::size()];
    kernel::tractionHistoryAverageBoundary avg;
    avg.n_q = fct[fctNo].get<Normal>().data()->data();
    avg.sigma_hist_q(0) = sigma0;
    avg.traction_hist_q = traction_hist_q;
    avg.execute();

    kernel::rhsFacetHistory rhs;
    rhs.b = B0.data();
    rhs.c3 = 1.0;
    rhs.traction_hist_q = traction_hist_q;
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.w = fctRule.weights().data();
    rhs.execute();
    return true;
}

bool PoissonViscoelasticity::rhs_traction_boundary(std::size_t fctNo, FacetInfo const& info,
                                                   Vector<double>& B0,
                                                   LinearAllocator<double>& scratch) const {
    if (info.bc != BC::Natural) {
        return false;
    }

    alignas(ALIGNMENT) double traction_q[tensor::traction_q::size()];
    auto t_q = Matrix<double>(traction_q, 1, tensor::traction_q::Shape[0]);
    fun_traction(fctNo, t_q, true);

    kernel::rhsTraction rhs;
    rhs.b = B0.data();
    rhs.traction_q = traction_q;
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.nl_q = fct[fctNo].get<NormalLength>().data();
    rhs.w = fctRule.weights().data();
    rhs.execute();
    return true;
}

// matrix free (mu_eff enters through volPre / fctPre)

void PoissonViscoelasticity::apply(std::size_t elNo, mneme::span<SideInfo> info,
                                   Vector<double const> const& x_0,
                                   std::array<Vector<double const>, NumFacets> const& x_n,
                                   Vector<double>& y_0) const {

    alignas(ALIGNMENT) double Dx_Q[tensor::Dx_Q::size()];
    kernel::apply_volume av;
    av.Dx_Q = Dx_Q;
    av.Dxi_Q = Dxi_Q.data();
    av.G_Q = vol[elNo].get<JInv>().data()->data();
    av.J_W_K_Q = volPre[elNo].get<AbsDetJWK>().data()->data();
    av.U = x_0.data();
    av.U_new = y_0.data();
    av.execute();

    alignas(ALIGNMENT) double n_q_flipped[tensor::n_q::size()];
    alignas(ALIGNMENT) double n_unit_q_flipped[tensor::n_unit_q::size()];
    for (std::size_t f = 0; f < NumFacets; ++f) {
        bool is_skeleton_face = elNo != info[f].lid;
        bool is_fault_or_dirichlet = info[f].bc == BC::Fault || info[f].bc == BC::Dirichlet;

        auto fctNo = info[f].fctNo;
        double const* n_q = fct[fctNo].get<Normal>().data()->data();
        double const* n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
        double const* K_G_q0 = fctPre[fctNo].get<KJInv0>().data()->data();
        double const* K_G_q1 = fctPre[fctNo].get<KJInv1>().data()->data();
        if (is_skeleton_face && info[f].side == 1) {
            std::swap(K_G_q0, K_G_q1);

            for (int i = 0; i < tensor::n_q::size(); ++i) {
                n_q_flipped[i] = -n_q[i];
            }
            n_q = n_q_flipped;
            for (int i = 0; i < tensor::n_unit_q::size(); ++i) {
                n_unit_q_flipped[i] = -n_unit_q[i];
            }
            n_unit_q = n_unit_q_flipped;
        }

        alignas(ALIGNMENT) double u_hat_q[tensor::u_hat_q::size()] = {};
        alignas(ALIGNMENT) double sigma_hat_q[tensor::sigma_hat_q::size()] = {};
        if (info[f].bc == BC::None || (is_skeleton_face && is_fault_or_dirichlet)) {
            kernel::flux_u_skeleton fu;
            fu.negative_E_q_T(0) = negative_E_q_T[f].data();
            fu.E_q_T(1) = E_q_T[info[f].localNo].data();
            fu.U = x_0.data();
            fu.U_ext = x_n[f].data();
            fu.u_hat_q = u_hat_q;
            fu.execute();

            kernel::flux_sigma_skeleton fs;
            fs.c00 = -penalty(fctNo);
            fs.Dxi_q_120(0) = Dxi_q_120[f].data();
            fs.Dxi_q_120(1) = Dxi_q_120[info[f].localNo].data();
            fs.E_q_T(0) = E_q_T[f].data();
            fs.negative_E_q_T(1) = negative_E_q_T[info[f].localNo].data();
            fs.K_G_q(0) = K_G_q0;
            fs.K_G_q(1) = K_G_q1;
            fs.U = x_0.data();
            fs.U_ext = x_n[f].data();
            fs.n_unit_q = n_unit_q;
            fs.sigma_hat_q = sigma_hat_q;
            fs.execute();
        } else if (is_fault_or_dirichlet) {
            kernel::flux_u_boundary fu;
            fu.U = x_0.data();
            fu.u_hat_q = u_hat_q;
            fu.negative_E_q_T(0) = negative_E_q_T[f].data();
            fu.execute();

            kernel::flux_sigma_boundary fs;
            fs.c00 = -penalty(fctNo);
            fs.Dxi_q_120(0) = Dxi_q_120[f].data();
            fs.E_q_T(0) = E_q_T[f].data();
            fs.K_G_q(0) = K_G_q0;
            fs.U = x_0.data();
            fs.n_unit_q = n_unit_q;
            fs.sigma_hat_q = sigma_hat_q;
            fs.execute();
        } else {
            continue;
        }

        kernel::apply_facet af;
        af.Dxi_q(0) = Dxi_q[f].data();
        af.E_q(0) = E_q[f].data();
        af.K_G_q(0) = K_G_q0;
        af.n_q = n_q;
        af.sigma_hat_q = sigma_hat_q;
        af.u_hat_q = u_hat_q;
        af.U_new = y_0.data();
        af.w = fctRule.weights().data();
        af.execute();
    }
}

std::size_t PoissonViscoelasticity::flops_apply(std::size_t elNo,
                                                mneme::span<SideInfo> info) const {
    std::size_t flops = kernel::apply_volume::HardwareFlops;
    for (std::size_t f = 0; f < NumFacets; ++f) {
        bool is_skeleton_face = elNo != info[f].lid;
        bool is_fault_or_dirichlet = info[f].bc == BC::Fault || info[f].bc == BC::Dirichlet;
        if (info[f].bc == BC::None || (is_skeleton_face && is_fault_or_dirichlet)) {
            flops += kernel::flux_u_skeleton::HardwareFlops;
            flops += kernel::flux_sigma_skeleton::HardwareFlops;
        } else if (is_fault_or_dirichlet) {
            flops += kernel::flux_u_boundary::HardwareFlops;
            flops += kernel::flux_sigma_boundary::HardwareFlops;
        } else {
            continue;
        }
        flops += kernel::apply_facet::HardwareFlops;
    }
    return flops;
}

// output

void PoissonViscoelasticity::coefficients_volume(std::size_t elNo, Matrix<double>& C,
                                                 LinearAllocator<double>&) const {
    auto const mu0 = material[elNo].get<K>();
    auto const mu1 = material[elNo].get<Mu1>();
    auto const tau = material[elNo].get<Tau>();
    auto const mu_eff = material[elNo].get<MuEff>();
    assert(mu0.size() == C.shape(0));
    for (std::size_t i = 0; i < mu0.size(); ++i) {
        C(i, 0) = mu0[i];
        C(i, 1) = mu1[i];
        C(i, 2) = tau[i];
        C(i, 3) = mu_eff[i];
    }
}

TensorBase<Matrix<double>> PoissonViscoelasticity::tractionResultInfo() const {
    return TensorBase<Matrix<double>>(tensor::grad_u::Shape[0], tensor::grad_u::Shape[1]);
}

void PoissonViscoelasticity::traction_skeleton(std::size_t fctNo, FacetInfo const& info,
                                               Vector<double const>& u0, Vector<double const>& u1,
                                               Matrix<double>& result) const {
    assert(result.size() == tensor::grad_u::size());

    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    bc_skeleton(fctNo, info.bc, f_q_raw);

    alignas(ALIGNMENT) double K_Dx_q0[tensor::K_Dx_q::size(0)];
    alignas(ALIGNMENT) double K_Dx_q1[tensor::K_Dx_q::size(1)];
    compute_K_Dx_q(fctNo, info, {K_Dx_q0, K_Dx_q1});

    kernel::grad_u krnl;
    krnl.c00 = -penalty(fctNo);
    krnl.K_Dx_q(0) = K_Dx_q0;
    krnl.K_Dx_q(1) = K_Dx_q1;
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.E_q(1) = E_q[info.localNo[1]].data();
    krnl.f_q = f_q_raw;
    krnl.grad_u = result.data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.u(0) = u0.data();
    krnl.u(1) = u1.data();
    krnl.execute();

    // Add {sigmaHat^n}: the friction law needs the TOTAL stress, not just
    // 2 mu_eff eps^{n+1}.
    alignas(ALIGNMENT) double sigma0[tensor::sigma_hist_q::size(0)];
    alignas(ALIGNMENT) double sigma1[tensor::sigma_hist_q::size(1)];
    compute_sigma_hist_q(fctNo, 0, sigma0);
    compute_sigma_hist_q(fctNo, 1, sigma1);

    kernel::grad_u_history hist;
    hist.grad_u = result.data();
    hist.sigma_hist_q(0) = sigma0;
    hist.sigma_hist_q(1) = sigma1;
    hist.execute();
}

void PoissonViscoelasticity::traction_boundary(std::size_t fctNo, FacetInfo const& info,
                                               Vector<double const>& u0,
                                               Matrix<double>& result) const {
    assert(result.size() == tensor::grad_u::size());

    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    bc_boundary(fctNo, info.bc, f_q_raw);

    alignas(ALIGNMENT) double K_Dx_q0[tensor::K_Dx_q::size(0)];
    compute_K_Dx_q(fctNo, info, {K_Dx_q0, nullptr});

    kernel::grad_u_bnd krnl;
    krnl.c00 = -penalty(fctNo);
    krnl.K_Dx_q(0) = K_Dx_q0;
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.f_q = f_q_raw;
    krnl.grad_u = result.data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.u(0) = u0.data();
    krnl.execute();

    alignas(ALIGNMENT) double sigma0[tensor::sigma_hist_q::size(0)];
    compute_sigma_hist_q(fctNo, 0, sigma0);

    kernel::grad_u_history_bnd hist;
    hist.grad_u = result.data();
    hist.sigma_hist_q(0) = sigma0;
    hist.execute();
}

} // namespace tndm