#include "Poisson.h"
#include "config.h"
#include "kernels/poisson/init.h"
#include "kernels/poisson/kernel.h"
#include "kernels/poisson/tensor.h"

#include "basis/WarpAndBlend.h"
#include "form/BC.h"
#include "form/DGCurvilinearCommon.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "tensor/EigenMap.h"
#include "util/LinearAllocator.h"
#include "util/Stopwatch.h"

#include <Eigen/Core>
#include <Eigen/LU>
#include <cassert>

namespace tensor = tndm::poisson::tensor;
namespace init = tndm::poisson::init;
namespace kernel = tndm::poisson::kernel;

namespace tndm {

Poisson::Poisson(std::shared_ptr<Curvilinear<DomainDimension>> cl, functional_t<1> K,
                 DGMethod method)
    : DGCurvilinearCommon<DomainDimension>(std::move(cl), MinQuadOrder()), method_(method),
      space_(PolynomialDegree, ALIGNMENT),
      materialSpace_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>(), ALIGNMENT),
      fun_K(make_volume_functional(std::move(K))), fun_force(zero_volume_function),
      fun_dirichlet(zero_facet_function), fun_slip(zero_facet_function) {

    Minv_ = space_.inverseMassMatrix();
    E_Q = space_.evaluateBasisAt(volRule.points());
    E_Q_T = space_.evaluateBasisAt(volRule.points(), {1, 0});
    Dxi_Q = space_.evaluateGradientAt(volRule.points());

    MinvRef_E_Q = Managed<Matrix<double>>(E_Q.shape(), std::size_t{ALIGNMENT});
    EigenMap(MinvRef_E_Q) = EigenMap(Minv_) * EigenMap(E_Q);

    MinvRef_E_Q_T = Managed<Matrix<double>>(E_Q_T.shape(), std::size_t{ALIGNMENT});
    EigenMap(MinvRef_E_Q_T) = EigenMap(E_Q_T) * EigenMap(Minv_).transpose();

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

void Poisson::compute_mass_matrix(std::size_t elNo, double* M) const {
    kernel::massMatrix mm;
    mm.E_Q = E_Q.data();
    mm.J_Q = vol[elNo].get<AbsDetJ>().data();
    mm.M = M;
    mm.W = volRule.weights().data();
    mm.execute();
}

void Poisson::compute_inverse_mass_matrix(std::size_t elNo, double* Minv) const {
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

void Poisson::compute_K_Dx_q(std::size_t fctNo, FacetInfo const& info,
                             std::array<double*, 2> K_Dx_q) const {
    kernel::K_Dx_q dx;
    for (int i = 0; i < 2; ++i) {
        if (K_Dx_q[i]) {
            auto JInv = (i == 1) ? fct[fctNo].get<JInv1>() : fct[fctNo].get<JInv0>();
            dx.G_q = JInv.data()->data();
            dx.matE_q_T = matE_q_T[info.localNo[i]].data();
            dx.K = material[info.up[i]].get<K>().data();
            dx.K_Dx_q(0) = K_Dx_q[i];
            dx.Dxi_q(0) = Dxi_q[info.localNo[i]].data();
            dx.execute();
        }
    }
}

void Poisson::compute_K_q(std::size_t fctNo, FacetInfo const& info,
                          std::array<double*, 2> K_q) const {
    kernel::K_q kw;
    for (int i = 0; i < 2; ++i) {
        if (K_q[i]) {
            kw.matE_q_T = matE_q_T[info.localNo[i]].data();
            kw.K = material[info.up[i]].get<K>().data();
            kw.K_q(0) = K_q[i];
            kw.execute();
        }
    }
}

void Poisson::begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                                std::size_t numLocalFacets) {
    base::begin_preparation(numElements, numLocalElements, numLocalFacets);

    material.setStorage(
        std::make_shared<material_vol_t>(numElements * materialSpace_.numBasisFunctions()), 0u,
        numElements, materialSpace_.numBasisFunctions());

    volPre.setStorage(std::make_shared<vol_pre_t>(numLocalElements * volRule.size()), 0u,
                      numLocalElements, volRule.size());

    const auto totalFacets = NumFacets * numElements;
    fct_on_vol_pre.setStorage(std::make_shared<fct_on_vol_pre_t>(totalFacets * fctRule.size()), 0u,
                              totalFacets, fctRule.size());
}

void Poisson::prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator<double>& scratch) {
    base::prepare_volume_post_skeleton(elNo, scratch);

    auto Kfield = material[elNo].get<K>().data();
    alignas(ALIGNMENT) double K_Q_raw[tensor::K_Q::size()];
    auto K_Q = Matrix<double>(K_Q_raw, 1, volRule.size());
    fun_K(elNo, K_Q);

    alignas(ALIGNMENT) double Mmem[tensor::matM::size()];
    kernel::project_K_lhs krnl_lhs;
    krnl_lhs.matE_Q_T = matE_Q_T.data();
    krnl_lhs.J_Q = vol[elNo].get<AbsDetJ>().data();
    krnl_lhs.matM = Mmem;
    krnl_lhs.W = volRule.weights().data();
    krnl_lhs.execute();

    kernel::project_K_rhs krnl_rhs;
    krnl_rhs.matE_Q_T = matE_Q_T.data();
    krnl_rhs.J_Q = vol[elNo].get<AbsDetJ>().data();
    krnl_rhs.K = Kfield;
    krnl_rhs.K_Q = K_Q_raw;
    krnl_rhs.W = volRule.weights().data();
    krnl_rhs.execute();

    using MMap = Eigen::Map<Eigen::Matrix<double, tensor::matM::Shape[0], tensor::matM::Shape[1]>,
                            Eigen::Unaligned,
                            Eigen::OuterStride<init::matM::Stop[0] - init::matM::Start[0]>>;
    using KMap = Eigen::Map<Eigen::Matrix<double, tensor::K::Shape[0], 1>, Eigen::Unaligned,
                            Eigen::InnerStride<1>>;

    auto K_eigen = KMap(Kfield);
    K_eigen = MMap(Mmem).fullPivLu().solve(K_eigen);

    auto Kmax = *std::max_element(Kfield, Kfield + materialSpace_.numBasisFunctions());
    base::penalty[elNo] *=
        Kmax * (PolynomialDegree + 1) * (PolynomialDegree + DomainDimension) / DomainDimension;

    kernel::J_W_G_Q krnl;
    krnl.G_Q = vol[elNo].get<JInv>().data()->data();
    krnl.J_W_G_Q = volPre[elNo].get<AbsDetJWJInv>().data()->data();
    krnl.J_Q = vol[elNo].get<AbsDetJ>().data();
    krnl.W = volRule.weights().data();
    krnl.execute();

    alignas(ALIGNMENT) double Jinv_Q[tensor::Jinv_Q::size()];
    auto J_Q = vol[elNo].get<AbsDetJ>();
    for (std::size_t q = 0; q < volRule.size(); ++q) {
        Jinv_Q[q] = 1.0 / J_Q[q];
    }
    kernel::K_W_Jinv_Q krnl2;
    krnl2.Jinv_Q = Jinv_Q;
    krnl2.K = material[elNo].get<K>().data();
    krnl2.K_W_Jinv_Q = volPre[elNo].get<KWAbsDetJInv>().data();
    krnl2.matE_Q_T = matE_Q_T.data();
    krnl2.W = volRule.weights().data();
    krnl2.execute();

    for (std::size_t f = 0; f < NumFacets; ++f) {
        auto idx = NumFacets * elNo + f;
        kernel::K_G_q k;
        k.G_q = fct_on_vol[idx].get<JInv0>().data()->data();
        k.K = material[elNo].get<K>().data();
        k.K_G_q(0) = fct_on_vol_pre[idx].get<KJInv>().data()->data();
        k.matE_q_T = matE_q_T[f].data();
        k.execute();
    }
}

bool Poisson::assemble_volume(std::size_t elNo, Matrix<double>& A00,
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
    krnl.K = material[elNo].get<K>().data();
    krnl.matE_Q_T = matE_Q_T.data();
    krnl.J_Q = vol[elNo].get<AbsDetJ>().data();
    krnl.W = volRule.weights().data();
    krnl.execute();
    return true;
}

bool Poisson::assemble_skeleton(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                                Matrix<double>& A01, Matrix<double>& A10, Matrix<double>& A11,
                                LinearAllocator<double>& scratch) const {
    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(E_q[0].shape(0) == tensor::E_q::Shape[0][0]);
    assert(E_q[0].shape(1) == tensor::E_q::Shape[0][1]);
    assert(Dxi_q[0].shape(0) == tensor::Dxi_q::Shape[0]);
    assert(Dxi_q[0].shape(1) == tensor::Dxi_q::Shape[1]);
    assert(Dxi_q[0].shape(2) == tensor::Dxi_q::Shape[2]);

    alignas(ALIGNMENT) double K_Dx_q0[tensor::K_Dx_q::size(0)];
    alignas(ALIGNMENT) double K_Dx_q1[tensor::K_Dx_q::size(1)];
    auto K_Dx_q = std::array<double*, 2>{K_Dx_q0, K_Dx_q1};
    compute_K_Dx_q(fctNo, info, K_Dx_q);

    alignas(ALIGNMENT) double L_q[2][std::max(tensor::L_q::size(0), tensor::L_q::size(1))];

    if (method_ == DGMethod::BR2) {
        alignas(ALIGNMENT) double Minv[2][tensor::M::size()];
        for (int i = 0; i < 2; ++i) {
            compute_inverse_mass_matrix(info.up[i], Minv[i]);
        }

        alignas(ALIGNMENT) double K_q0[tensor::K_q::size(0)];
        alignas(ALIGNMENT) double K_q1[tensor::K_q::size(1)];
        auto K_q = std::array<double*, 2>{K_q0, K_q1};
        compute_K_q(fctNo, info, K_q);

        kernel::lift_skeleton lift;
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
    assemble.c20 = penalty(info);
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

bool Poisson::assemble_boundary(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                                LinearAllocator<double>& scratch) const {
    if (info.bc == BC::Natural) {
        return false;
    }

    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(E_q[0].shape(0) == tensor::E_q::Shape[0][0]);
    assert(E_q[0].shape(1) == tensor::E_q::Shape[0][1]);
    assert(Dxi_q[0].shape(0) == tensor::Dxi_q::Shape[0]);
    assert(Dxi_q[0].shape(1) == tensor::Dxi_q::Shape[1]);
    assert(Dxi_q[0].shape(2) == tensor::Dxi_q::Shape[2]);

    alignas(ALIGNMENT) double L0[tensor::L_q::size(0)];
    if (method_ == DGMethod::BR2) {
        alignas(ALIGNMENT) double Minv0[tensor::M::size()];
        compute_inverse_mass_matrix(info.up[0], Minv0);

        alignas(ALIGNMENT) double K_q[tensor::K_q::size(0)];
        compute_K_q(fctNo, info, {K_q, nullptr});

        kernel::lift_boundary lift;
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
    assemble.c20 = penalty(info);
    assemble.a(0, 0) = A00.data();
    assemble.K_Dx_q(0) = K_Dx_q0;
    assemble.E_q(0) = E_q[info.localNo[0]].data();
    assemble.L_q(0) = L0;
    assemble.n_q = fct[fctNo].get<Normal>().data()->data();
    assemble.w = fctRule.weights().data();
    assemble.execute(0, 0);
    return true;
}

bool Poisson::rhs_volume(std::size_t elNo, Vector<double>& B,
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

bool Poisson::bc_skeleton(std::size_t fctNo, BC bc, double f_q_raw[]) const {
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
bool Poisson::bc_boundary(std::size_t fctNo, BC bc, double f_q_raw[]) const {
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

bool Poisson::rhs_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                           Vector<double>& B1, LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    if (!bc_skeleton(fctNo, info.bc, f_q_raw)) {
        return false;
    }

    alignas(ALIGNMENT) double f_lifted_q[tensor::f_lifted_q::size()];
    if (method_ == DGMethod::BR2) {
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
    rhs.c20 = penalty(info);
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

bool Poisson::rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                           LinearAllocator<double>& scratch) const {
    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    if (!bc_boundary(fctNo, info.bc, f_q_raw)) {
        return false;
    }

    alignas(ALIGNMENT) double f_lifted_q[tensor::f_lifted_q::size()];
    if (method_ == DGMethod::BR2) {
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
    rhs.c20 = penalty(info);
    rhs.f_q = f_q_raw;
    rhs.f_lifted_q = f_lifted_q;
    rhs.n_q = fct[fctNo].get<Normal>().data()->data();
    rhs.w = fctRule.weights().data();
    rhs.K_Dx_q(0) = K_Dx_q0;
    rhs.E_q(0) = E_q[info.localNo[0]].data();
    rhs.execute();
    return true;
}

void Poisson::apply(std::size_t elNo, mneme::span<SideInfo> info, Vector<double const> const& x_0,
                    std::array<Vector<double const>, NumFacets> const& x_n,
                    Vector<double>& y_0) const {
    Stopwatch sw;
    sw.start();

    alignas(ALIGNMENT) double sigma[tensor::sigma::size()];
    kernel::stress_volume sv;
    sv.Dxi_Q = Dxi_Q.data();
    sv.J_W_G_Q = volPre[elNo].get<AbsDetJWJInv>().data()->data();
    sv.negative_E_Q_T = negative_E_Q_T.data();
    sv.U = x_0.data();
    sv.sigma = sigma;
    sv.execute();
    auto sv_time = sw.stop();
    auto sv_flops = kernel::stress_volume::HardwareFlops;

    unsigned sf_flops = 0;
    sw.start();
    alignas(ALIGNMENT) double u_hat_q[tensor::u_hat_q::size()];
    for (std::size_t f = 0; f < NumFacets; ++f) {
        if (info[f].bc == BC::None || info[f].bc == BC::Fault) {
            kernel::flux_u_skeleton fu;
            fu.E_q_T(0) = E_q_T[f].data();
            fu.E_q_T(1) = E_q_T[info[f].localNo].data();
            fu.U = x_0.data();
            fu.U_ext = x_n[f].data();
            fu.u_hat_q = u_hat_q;
            fu.execute();
            sf_flops += kernel::flux_u_skeleton::HardwareFlops;
        } else if (info[f].bc == BC::Natural) {
            kernel::flux_u_boundary fu;
            fu.U = x_0.data();
            fu.u_hat_q = u_hat_q;
            fu.E_q_T(0) = E_q_T[f].data();
            fu.execute();
        } else {
            continue;
        }

        kernel::stress_facet sf;
        sf.sigma = sigma;
        sf.u_hat_q = u_hat_q;
        sf.w = fctRule.weights().data();
        sf.E_q(0) = E_q[f].data();
        sf.n_q = fct_on_vol[NumFacets * elNo + f].get<Normal>().data()->data();
        sf.execute();
        sf_flops += kernel::stress_facet::HardwareFlops;
    }
    auto sf_time = sw.stop();

    unsigned ps_flops = 0;
    sw.start();
    kernel::project_stress ps;
    ps.K_W_Jinv_Q = volPre[elNo].get<KWAbsDetJInv>().data();
    ps.MinvRef_E_Q = MinvRef_E_Q.data();
    ps.MinvRef_E_Q_T = MinvRef_E_Q_T.data();
    ps.sigma = sigma;
    ps.execute();
    ps_flops += kernel::project_stress::HardwareFlops;

    auto ps_time = sw.stop();

    unsigned av_flops = 0;
    sw.start();
    kernel::apply_volume av;
    av.Dxi_Q = Dxi_Q.data();
    av.E_Q = E_Q.data();
    av.J_W_G_Q = volPre[elNo].get<AbsDetJWJInv>().data()->data();
    av.U_new = y_0.data();
    av.sigma = sigma;
    av.execute();
    av_flops += kernel::apply_volume::HardwareFlops;

    auto av_time = sw.stop();

    unsigned af_flops = 0;
    sw.start();
    alignas(ALIGNMENT) double sigma_hat_q[tensor::sigma_hat_q::size()];
    for (std::size_t f = 0; f < NumFacets; ++f) {
        if (info[f].bc == BC::None || info[f].bc == BC::Fault) {
            kernel::flux_sigma_skeleton fs;
            fs.c00 = -penalty(elNo, info[f].lid);
            fs.Dxi_q_120(0) = Dxi_q_120[f].data();
            fs.Dxi_q_120(1) = Dxi_q_120[info[f].localNo].data();
            fs.E_q_T(0) = E_q_T[f].data();
            fs.negative_E_q_T(1) = negative_E_q_T[info[f].localNo].data();
            fs.K_G_q(0) = fct_on_vol_pre[NumFacets * elNo + f].get<KJInv>().data()->data();
            fs.K_G_q(1) = fct_on_vol_pre[NumFacets * info[f].lid + info[f].localNo]
                              .get<KJInv>()
                              .data()
                              ->data();
            fs.U = x_0.data();
            fs.U_ext = x_n[f].data();
            fs.n_unit_q = fct_on_vol[NumFacets * elNo + f].get<UnitNormal>().data()->data();
            fs.sigma_hat_q = sigma_hat_q;
            fs.execute();
            af_flops += kernel::flux_sigma_skeleton::HardwareFlops;
        } else if (info[f].bc == BC::Dirichlet) {
            kernel::flux_sigma_boundary fs;
            fs.c00 = -penalty(elNo, elNo);
            fs.Dxi_q_120(0) = Dxi_q_120[f].data();
            fs.E_q_T(0) = E_q_T[f].data();
            fs.K_G_q(0) = fct_on_vol_pre[NumFacets * elNo + f].get<KJInv>().data()->data();
            fs.U = x_0.data();
            fs.n_unit_q = fct_on_vol[NumFacets * elNo + f].get<UnitNormal>().data()->data();
            fs.sigma_hat_q = sigma_hat_q;
            fs.execute();
            af_flops += kernel::flux_sigma_boundary::HardwareFlops;
        } else {
            continue;
        }

        kernel::apply_facet af;
        af.E_q(0) = E_q[f].data();
        af.U_new = y_0.data();
        af.n_q = fct_on_vol[NumFacets * elNo + f].get<Normal>().data()->data();
        af.sigma_hat_q = sigma_hat_q;
        af.w = fctRule.weights().data();
        af.execute();
        af_flops += kernel::apply_facet::HardwareFlops;
    }
    auto af_time = sw.stop();

    // std::cout << sv_time << " " << sf_time << " " << ps_time << " " << av_time << " " << af_time
    //<< std::endl;
    // double time = (sv_time + sf_time + ps_time + av_time + af_time);
    // std::cout << sv_time / time * 100 << " " << sf_time / time * 100 << " " << ps_time / time *
    // 100
    //<< " " << av_time / time * 100 << " " << af_time / time * 100 << std::endl;
    // std::cout << sv_flops << " " << sf_flops << " " << ps_flops << " " << av_flops << " "
    //<< af_flops << std::endl;
    // std::cout << sv_flops / sv_time * 1e-9 << " " << sf_flops / sf_time * 1e-9 << " "
    //<< ps_flops / ps_time * 1e-9 << " " << av_flops / av_time * 1e-9 << " "
    //<< af_flops / af_time * 1e-9 << std::endl;
    // unsigned flops = sv_flops + sf_flops + ps_flops + av_flops + af_flops;
    // std::cout << flops << " " << flops / time / 1e9 << std::endl;
}

void Poisson::coefficients_volume(std::size_t elNo, Matrix<double>& C,
                                  LinearAllocator<double>&) const {
    auto const coeff_K = material[elNo].get<K>();
    assert(coeff_K.size() == C.shape(0));
    for (std::size_t i = 0; i < coeff_K.size(); ++i) {
        C(i, 0) = coeff_K[i];
    }
}

TensorBase<Matrix<double>> Poisson::tractionResultInfo() const {
    return TensorBase<Matrix<double>>(tensor::grad_u::Shape[0], tensor::grad_u::Shape[1]);
}

void Poisson::traction_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                                Vector<double const>& u1, Matrix<double>& result) const {
    assert(result.size() == tensor::grad_u::size());

    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    bc_skeleton(fctNo, info.bc, f_q_raw);

    alignas(ALIGNMENT) double K_Dx_q0[tensor::K_Dx_q::size(0)];
    alignas(ALIGNMENT) double K_Dx_q1[tensor::K_Dx_q::size(1)];
    compute_K_Dx_q(fctNo, info, {K_Dx_q0, K_Dx_q1});

    kernel::grad_u krnl;
    krnl.c00 = -penalty(info);
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
}

void Poisson::traction_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                                Matrix<double>& result) const {
    assert(result.size() == tensor::grad_u::size());

    alignas(ALIGNMENT) double f_q_raw[tensor::f_q::size()];
    bc_boundary(fctNo, info.bc, f_q_raw);

    alignas(ALIGNMENT) double K_Dx_q0[tensor::K_Dx_q::size(0)];
    compute_K_Dx_q(fctNo, info, {K_Dx_q0, nullptr});

    kernel::grad_u_bnd krnl;
    krnl.c00 = -penalty(info);
    krnl.K_Dx_q(0) = K_Dx_q0;
    krnl.E_q(0) = E_q[info.localNo[0]].data();
    krnl.f_q = f_q_raw;
    krnl.grad_u = result.data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.u(0) = u0.data();
    krnl.execute();
}

} // namespace tndm
