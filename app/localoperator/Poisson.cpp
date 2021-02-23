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
#include "util/LinearAllocator.h"

#include <Eigen/LU>
#include <cassert>

namespace tensor = tndm::poisson::tensor;
namespace init = tndm::poisson::init;
namespace kernel = tndm::poisson::kernel;

namespace tndm {

Poisson::Poisson(std::shared_ptr<Curvilinear<DomainDimension>> cl, functional_t<1> K)
    : DGCurvilinearCommon<DomainDimension>(std::move(cl), MinQuadOrder()), space_(PolynomialDegree),
      materialSpace_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>()),
      fun_K(make_volume_functional(std::move(K))), fun_force(zero_volume_function),
      fun_dirichlet(zero_facet_function), fun_slip(zero_facet_function) {

    E_Q = space_.evaluateBasisAt(volRule.points());
    Dxi_Q = space_.evaluateGradientAt(volRule.points());
    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto points = cl_->facetParam(f, fctRule.points());
        E_q.emplace_back(space_.evaluateBasisAt(points));
        Dxi_q.emplace_back(space_.evaluateGradientAt(points));
        matE_q_T.emplace_back(materialSpace_.evaluateBasisAt(points, {1, 0}));
    }

    matE_Q_T = materialSpace_.evaluateBasisAt(volRule.points(), {1, 0});
}

void Poisson::compute_mass_matrix(std::size_t elNo, double* M) const {
    kernel::massMatrix mm;
    mm.E = E_Q.data();
    mm.J = vol[elNo].get<AbsDetJ>().data();
    mm.M = M;
    mm.W = volRule.weights().data();
    mm.execute();
}

void Poisson::compute_inverse_mass_matrix(std::size_t elNo, double* Minv) const {
    compute_mass_matrix(elNo, Minv);

    using MMat = Eigen::Matrix<double, tensor::M::Shape[0], tensor::M::Shape[1]>;
    using MMap = Eigen::Map<MMat, Eigen::Unaligned,
                            Eigen::OuterStride<init::M::Stop[0] - init::M::Start[0]>>;
    auto Minv_eigen = MMap(Minv);
    auto Minv_lu = Eigen::FullPivLU<MMat>(Minv_eigen);
    Minv_eigen = Minv_lu.inverse();
}

void Poisson::begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                                std::size_t numLocalFacets) {
    base::begin_preparation(numElements, numLocalElements, numLocalFacets);

    material.setStorage(
        std::make_shared<material_vol_t>(numElements * materialSpace_.numBasisFunctions()), 0u,
        numElements, materialSpace_.numBasisFunctions());
}

void Poisson::prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator<double>& scratch) {
    base::prepare_volume_post_skeleton(elNo, scratch);

    auto Kfield = material[elNo].get<K>().data();
    double K_Q_raw[tensor::K_Q::size()];
    auto K_Q = Matrix<double>(K_Q_raw, 1, volRule.size());
    fun_K(elNo, K_Q);

    double Mmem[tensor::matM::size()];
    kernel::project_K_lhs krnl_lhs;
    krnl_lhs.Em = matE_Q_T.data();
    krnl_lhs.J = vol[elNo].get<AbsDetJ>().data();
    krnl_lhs.matM = Mmem;
    krnl_lhs.W = volRule.weights().data();
    krnl_lhs.execute();

    kernel::project_K_rhs krnl_rhs;
    krnl_rhs.Em = matE_Q_T.data();
    krnl_rhs.J = vol[elNo].get<AbsDetJ>().data();
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
}

bool Poisson::assemble_volume(std::size_t elNo, Matrix<double>& A00,
                              LinearAllocator<double>& scratch) const {
    double Dx_Q[tensor::D_x::size()];

    assert(volRule.size() == tensor::W::Shape[0]);
    assert(Dxi_Q.shape(0) == tensor::D_xi::Shape[0]);
    assert(Dxi_Q.shape(1) == tensor::D_xi::Shape[1]);
    assert(Dxi_Q.shape(2) == tensor::D_xi::Shape[2]);

    kernel::assembleVolume krnl;
    krnl.A = A00.data();
    krnl.D_x = Dx_Q;
    krnl.D_xi = Dxi_Q.data();
    krnl.G = vol[elNo].get<JInv>().data()->data();
    krnl.K = material[elNo].get<K>().data();
    krnl.Em = matE_Q_T.data();
    krnl.J = vol[elNo].get<AbsDetJ>().data();
    krnl.W = volRule.weights().data();
    krnl.execute();
    return true;
}

bool Poisson::assemble_skeleton(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                                Matrix<double>& A01, Matrix<double>& A10, Matrix<double>& A11,
                                LinearAllocator<double>& scratch) const {
    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(E_q[0].shape(0) == tensor::e::Shape[0][0]);
    assert(E_q[0].shape(1) == tensor::e::Shape[0][1]);
    assert(Dxi_q[0].shape(0) == tensor::d_xi::Shape[0][0]);
    assert(Dxi_q[0].shape(1) == tensor::d_xi::Shape[0][1]);
    assert(Dxi_q[0].shape(2) == tensor::d_xi::Shape[0][2]);

    double Dx_q0[tensor::d_x::size(0)];
    double Dx_q1[tensor::d_x::size(1)];

    double Minv[2][tensor::M::size()];
    for (int i = 0; i < 2; ++i) {
        compute_inverse_mass_matrix(info.up[i], Minv[i]);
    }

    double K_w_q[2][tensor::K_w_q::size(0)];
    kernel::Kw kw;
    kw.w = fctRule.weights().data();
    for (int i = 0; i < 2; ++i) {
        kw.K = material[info.up[i]].get<K>().data();
        kw.K_w_q(i) = K_w_q[i];
        kw.em(i) = matE_q_T[info.localNo[i]].data();
        kw.execute(i);
    }

    double L_q[2][std::max(tensor::L_q::size(0), tensor::L_q::size(1))];
    kernel::lift_skeleton lift;
    lift.n = fct[fctNo].get<Normal>().data()->data();
    for (int i = 0; i < 2; ++i) {
        lift.K_w_q(i) = K_w_q[i];
        lift.L_q(i) = L_q[i];
        lift.Minv(i) = Minv[i];
        lift.e(i) = E_q[info.localNo[i]].data();
    }
    lift.execute(0);
    lift.execute(1);

    kernel::assembleFacetLocal local;
    local.c00 = -0.5;
    local.c10 = epsilon * 0.5;
    local.c20 = penalty(info) * 0.5;
    local.a(0, 0) = A00.data();
    local.d_x(0) = Dx_q0;
    local.d_xi(0) = Dxi_q[info.localNo[0]].data();
    local.e(0) = E_q[info.localNo[0]].data();
    local.em(0) = matE_q_T[info.localNo[0]].data();
    local.g(0) = fct[fctNo].get<JInv0>().data()->data();
    local.K = material[info.up[0]].get<K>().data();
    local.L_q(0) = L_q[0];
    local.n = fct[fctNo].get<Normal>().data()->data();
    // local.nl = fct[fctNo].get<NormalLength>().data();
    local.w = fctRule.weights().data();
    local.execute();

    kernel::assembleFacetNeighbour neighbour;
    neighbour.c00 = local.c00;
    neighbour.c01 = -local.c00;
    neighbour.c10 = local.c10;
    neighbour.c11 = -local.c10;
    neighbour.c20 = local.c20;
    neighbour.c21 = -local.c20;
    neighbour.a(0, 1) = A01.data();
    neighbour.a(1, 0) = A10.data();
    neighbour.a(1, 1) = A11.data();
    neighbour.d_x(0) = Dx_q0;
    neighbour.d_x(1) = Dx_q1;
    neighbour.d_xi(1) = Dxi_q[info.localNo[1]].data();
    neighbour.e(0) = local.e(0);
    neighbour.em(0) = local.em(0);
    neighbour.e(1) = E_q[info.localNo[1]].data();
    neighbour.em(1) = matE_q_T[info.localNo[1]].data();
    neighbour.g(1) = fct[fctNo].get<JInv1>().data()->data();
    neighbour.K = material[info.up[1]].get<K>().data();
    neighbour.L_q(0) = L_q[0];
    neighbour.L_q(1) = L_q[1];
    neighbour.n = local.n;
    // neighbour.nl = local.nl;
    neighbour.w = local.w;
    neighbour.execute();
    return true;
}

bool Poisson::assemble_boundary(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                                LinearAllocator<double>& scratch) const {
    if (info.bc == BC::Natural) {
        return false;
    }

    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(E_q[0].shape(0) == tensor::e::Shape[0][0]);
    assert(E_q[0].shape(1) == tensor::e::Shape[0][1]);
    assert(Dxi_q[0].shape(0) == tensor::d_xi::Shape[0][0]);
    assert(Dxi_q[0].shape(1) == tensor::d_xi::Shape[0][1]);
    assert(Dxi_q[0].shape(2) == tensor::d_xi::Shape[0][2]);

    double Minv0[tensor::M::size()];
    compute_inverse_mass_matrix(info.up[0], Minv0);

    double K_w_q[tensor::K_w_q::size(0)];
    kernel::Kw kw;
    kw.w = fctRule.weights().data();
    kw.K = material[info.up[0]].get<K>().data();
    kw.K_w_q(0) = K_w_q;
    kw.em(0) = matE_q_T[info.localNo[0]].data();
    kw.execute(0);

    double L0[tensor::L_q::size(0)];
    kernel::lift_boundary lift;
    lift.K_w_q(0) = K_w_q;
    lift.L_q(0) = L0;
    lift.Minv(0) = Minv0;
    lift.e(0) = E_q[info.localNo[0]].data();
    lift.n = fct[fctNo].get<Normal>().data()->data();
    lift.execute();

    double Dx_q0[tensor::d_x::size(0)];

    kernel::assembleFacetLocal local;
    local.c00 = -1.0;
    local.c10 = epsilon;
    local.c20 = penalty(info);
    local.a(0, 0) = A00.data();
    local.d_x(0) = Dx_q0;
    local.d_xi(0) = Dxi_q[info.localNo[0]].data();
    local.e(0) = E_q[info.localNo[0]].data();
    local.em(0) = matE_q_T[info.localNo[0]].data();
    local.g(0) = fct[fctNo].get<JInv0>().data()->data();
    local.K = material[info.up[0]].get<K>().data();
    local.L_q(0) = L0;
    local.n = fct[fctNo].get<Normal>().data()->data();
    // local.nl = fct[fctNo].get<NormalLength>().data();
    local.w = fctRule.weights().data();
    local.execute();
    return true;
}

bool Poisson::rhs_volume(std::size_t elNo, Vector<double>& B,
                         LinearAllocator<double>& scratch) const {
    assert(tensor::b::Shape[0] == tensor::A::Shape[0]);

    double F_Q_raw[tensor::F_Q::size()];
    assert(tensor::F_Q::size() == volRule.size());
    auto F_Q = Matrix<double>(F_Q_raw, 1, tensor::F_Q::Shape[0]);
    fun_force(elNo, F_Q);

    kernel::rhsVolume rhs;
    rhs.E = E_Q.data();
    rhs.F_Q = F_Q_raw;
    rhs.J = vol[elNo].get<AbsDetJ>().data();
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
    double f_q_raw[tensor::f_q::size()];
    if (!bc_skeleton(fctNo, info.bc, f_q_raw)) {
        return false;
    }

    double Minv[2][tensor::M::size()];
    compute_inverse_mass_matrix(info.up[0], Minv[0]);
    compute_inverse_mass_matrix(info.up[1], Minv[1]);

    double K_w_q[2][tensor::K_w_q::size(0)];
    kernel::Kw kw;
    kw.w = fctRule.weights().data();
    for (int i = 0; i < 2; ++i) {
        kw.K = material[info.up[i]].get<K>().data();
        kw.K_w_q(i) = K_w_q[i];
        kw.em(i) = matE_q_T[info.localNo[i]].data();
        kw.execute(i);
    }

    double f_lifted[tensor::f_lifted::size()];
    kernel::rhs_lift_skeleton lift;
    for (int i = 0; i < 2; ++i) {
        lift.e(i) = E_q[info.localNo[i]].data();
        lift.K_w_q(i) = K_w_q[i];
        lift.Minv(i) = Minv[i];
    }
    lift.n = fct[fctNo].get<Normal>().data()->data();
    lift.f_q = f_q_raw;
    lift.f_lifted = f_lifted;
    lift.execute();

    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = 0.5 * epsilon;
    rhs.c20 = penalty(info);
    rhs.f_q = f_q_raw;
    rhs.f_lifted = f_lifted;
    rhs.n = fct[fctNo].get<Normal>().data()->data();
    // rhs.nl = fct[fctNo].get<NormalLength>().data();
    rhs.w = fctRule.weights().data();
    rhs.d_xi(0) = Dxi_q[info.localNo[0]].data();
    rhs.e(0) = E_q[info.localNo[0]].data();
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.K_w_q(0) = K_w_q[0];
    rhs.execute();

    rhs.b = B1.data();
    rhs.c20 *= -1.0;
    rhs.d_xi(0) = Dxi_q[info.localNo[1]].data();
    rhs.e(0) = E_q[info.localNo[1]].data();
    rhs.g(0) = fct[fctNo].get<JInv1>().data()->data();
    rhs.K_w_q(0) = K_w_q[1];
    rhs.execute();

    return true;
}

bool Poisson::rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                           LinearAllocator<double>& scratch) const {
    double f_q_raw[tensor::f_q::size()];
    if (!bc_boundary(fctNo, info.bc, f_q_raw)) {
        return false;
    }

    double M0[tensor::M::size()];
    compute_inverse_mass_matrix(info.up[0], M0);

    double K_w_q[tensor::K_w_q::size(0)];
    kernel::Kw kw;
    kw.w = fctRule.weights().data();
    kw.K = material[info.up[0]].get<K>().data();
    kw.K_w_q(0) = K_w_q;
    kw.em(0) = matE_q_T[info.localNo[0]].data();
    kw.execute(0);

    double f_lifted[tensor::f_lifted::size()];
    kernel::rhs_lift_boundary lift;
    lift.e(0) = E_q[info.localNo[0]].data();
    lift.n = fct[fctNo].get<Normal>().data()->data();
    lift.K_w_q(0) = K_w_q;
    lift.Minv(0) = M0;
    lift.f_q = f_q_raw;
    lift.f_lifted = f_lifted;
    lift.execute();

    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = epsilon;
    rhs.c20 = penalty(info);
    rhs.f_q = f_q_raw;
    rhs.f_lifted = f_lifted;
    rhs.n = fct[fctNo].get<Normal>().data()->data();
    // rhs.nl = fct[fctNo].get<NormalLength>().data();
    rhs.w = fctRule.weights().data();
    rhs.d_xi(0) = Dxi_q[info.localNo[0]].data();
    rhs.e(0) = E_q[info.localNo[0]].data();
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.K_w_q(0) = K_w_q;
    rhs.execute();
    return true;
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

    double Dx_q0[tensor::d_x::size(0)];
    double Dx_q1[tensor::d_x::size(1)];
    double f_q_raw[tensor::f_q::size()];
    bc_skeleton(fctNo, info.bc, f_q_raw);

    kernel::grad_u krnl;
    krnl.c00 = -penalty(info);
    krnl.d_x(0) = Dx_q0;
    krnl.d_x(1) = Dx_q1;
    for (std::size_t side = 0; side < 2; ++side) {
        krnl.d_xi(side) = Dxi_q[info.localNo[side]].data();
        krnl.em(side) = matE_q_T[info.localNo[side]].data();
    }
    krnl.e(0) = E_q[info.localNo[0]].data();
    krnl.e(1) = E_q[info.localNo[1]].data();
    krnl.f_q = f_q_raw;
    krnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    krnl.g(1) = fct[fctNo].get<JInv1>().data()->data();
    krnl.grad_u = result.data();
    krnl.k(0) = material[info.up[0]].get<K>().data();
    krnl.k(1) = material[info.up[1]].get<K>().data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.u(0) = u0.data();
    krnl.u(1) = u1.data();
    krnl.execute();
}

void Poisson::traction_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                                Matrix<double>& result) const {
    assert(result.size() == tensor::grad_u::size());

    double Dx_q0[tensor::d_x::size(0)];
    double f_q_raw[tensor::f_q::size()];
    bc_boundary(fctNo, info.bc, f_q_raw);

    kernel::grad_u_bnd krnl;
    krnl.c00 = -penalty(info);
    krnl.d_x(0) = Dx_q0;
    krnl.d_xi(0) = Dxi_q[info.localNo[0]].data();
    krnl.em(0) = matE_q_T[info.localNo[0]].data();
    krnl.e(0) = E_q[info.localNo[0]].data();
    krnl.f_q = f_q_raw;
    krnl.g(0) = fct[fctNo].get<JInv0>().data()->data();
    krnl.grad_u = result.data();
    krnl.k(0) = material[info.up[0]].get<K>().data();
    krnl.n_unit_q = fct[fctNo].get<UnitNormal>().data()->data();
    krnl.u(0) = u0.data();
    krnl.execute();
}

} // namespace tndm
