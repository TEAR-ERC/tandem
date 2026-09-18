#include "Poisson.h"
#include "config.h"
#include "kernels/poisson/init.h"
#include "kernels/poisson/kernel.h"
#include "kernels/poisson/tensor.h"

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

    volPre.setStorage(std::make_shared<vol_pre_t>(numElements * volRule.size()), 0u,
                      numLocalElements, volRule.size());

    fctPre.setStorage(std::make_shared<fct_pre_t>(numLocalFacets * fctRule.size()), 0u,
                      numLocalFacets, fctRule.size());

    penalty_.resize(numLocalFacets);
}

void Poisson::prepare_volume(std::size_t elNo, LinearAllocator<double>& scratch) {
    base::prepare_volume(elNo, scratch);

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
}

void Poisson::prepare_skeleton(std::size_t fctNo, FacetInfo const& info,
                               LinearAllocator<double>& scratch) {
    base::prepare_skeleton(fctNo, info, scratch);

    for (int side = 0; side < 2; ++side) {
        kernel::K_G_q k;
        k.G_q = side == 1 ? fct[fctNo].get<JInv1>().data()->data()
                          : fct[fctNo].get<JInv0>().data()->data();
        k.K = material[info.up[side]].get<K>().data();
        k.K_G_q(0) = side == 1 ? fctPre[fctNo].get<KJInv1>().data()->data()
                               : fctPre[fctNo].get<KJInv0>().data()->data();
        k.matE_q_T = matE_q_T[info.localNo[side]].data();
        k.execute();
    }
}

void Poisson::prepare_boundary(std::size_t fctNo, FacetInfo const& info,
                               LinearAllocator<double>& scratch) {
    base::prepare_boundary(fctNo, info, scratch);

    kernel::K_G_q k;
    k.G_q = fct[fctNo].get<JInv0>().data()->data();
    k.K = material[info.up[0]].get<K>().data();
    k.K_G_q(0) = fctPre[fctNo].get<KJInv0>().data()->data();
    k.matE_q_T = matE_q_T[info.localNo[0]].data();
    k.execute();
}

void Poisson::prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator<double>& scratch) {
    base::prepare_volume_post_skeleton(elNo, scratch);

    auto Kfield = material[elNo].get<K>().data();

    kernel::J_W_K_Q krnl;
    krnl.J_W_K_Q = volPre[elNo].get<AbsDetJWK>().data()->data();
    krnl.J_Q = vol[elNo].get<AbsDetJ>().data();
    krnl.K = Kfield;
    krnl.matE_Q_T = matE_Q_T.data();
    krnl.W = volRule.weights().data();
    krnl.execute();
}

void Poisson::prepare_penalty(std::size_t fctNo, FacetInfo const& info, LinearAllocator<double>&) {
    auto const p = [&](int side) {
        auto Kfield = material[info.up[side]].get<K>().data();
        auto k0 = *std::min_element(Kfield, Kfield + materialSpace_.numBasisFunctions());
        auto k1 = *std::max_element(Kfield, Kfield + materialSpace_.numBasisFunctions());
        constexpr double c_N_1 = InverseInequality<Dim>::trace_constant(PolynomialDegree - 1);
        return (Dim + 1) * c_N_1 * (area_[fctNo] / volume_[info.up[side]]) * (k1 * k1 / k0);
    };

    if (info.up[0] != info.up[1]) {
        penalty_[fctNo] = (p(0) + p(1)) / 4.0;
    } else {
        penalty_[fctNo] = p(0);
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
    assert(Dxi_q[0].shape(0) == tensor::Dxi_q::Shape[0][0]);
    assert(Dxi_q[0].shape(1) == tensor::Dxi_q::Shape[0][1]);
    assert(Dxi_q[0].shape(2) == tensor::Dxi_q::Shape[0][2]);

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

bool Poisson::assemble_boundary(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                                LinearAllocator<double>& scratch) const {
    if (info.bc == BC::Natural) {
        return false;
    }

    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(E_q[0].shape(0) == tensor::E_q::Shape[0][0]);
    assert(E_q[0].shape(1) == tensor::E_q::Shape[0][1]);
    assert(Dxi_q[0].shape(0) == tensor::Dxi_q::Shape[0][0]);
    assert(Dxi_q[0].shape(1) == tensor::Dxi_q::Shape[0][1]);
    assert(Dxi_q[0].shape(2) == tensor::Dxi_q::Shape[0][2]);

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

bool Poisson::rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
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

void Poisson::apply(std::size_t elNo, mneme::span<SideInfo> info, Vector<double const> const& x_0,
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

std::size_t Poisson::flops_apply(std::size_t elNo, mneme::span<SideInfo> info) const {
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
}

void Poisson::traction_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
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
}
double* Poisson::get_mu_field(FacetInfo const& info) const {
    double* a;
    return a;
}
} // namespace tndm
