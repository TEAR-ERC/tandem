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

Poisson::Poisson(std::shared_ptr<Curvilinear<DomainDimension>> cl, functional_t<1> K,
                 DGMethod method)
    : DGCurvilinearCommon<DomainDimension>(std::move(cl), MinQuadOrder()), method_(method),
      space_(PolynomialDegree),
      materialSpace_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>()),
      fun_K(make_volume_functional(std::move(K))), fun_force(zero_volume_function),
      fun_dirichlet(zero_facet_function), fun_slip(zero_facet_function) {

    Minv_ = space_.inverseMassMatrix();
    E_Q = space_.evaluateBasisAt(volRule.points());
    Dxi_Q = space_.evaluateGradientAt(volRule.points());
    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto points = cl_->facetParam(f, fctRule.points());
        E_q.emplace_back(space_.evaluateBasisAt(points));
        Dxi_q.emplace_back(space_.evaluateGradientAt(points));
        matE_q_T.emplace_back(materialSpace_.evaluateBasisAt(points, {1, 0}));
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
    double Jinv_Q[tensor::Jinv_Q::size()] = {};
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
            dx.Dxi_q = Dxi_q[info.localNo[i]].data();
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
}

void Poisson::prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator<double>& scratch) {
    base::prepare_volume_post_skeleton(elNo, scratch);

    auto Kfield = material[elNo].get<K>().data();
    double K_Q_raw[tensor::K_Q::size()];
    auto K_Q = Matrix<double>(K_Q_raw, 1, volRule.size());
    fun_K(elNo, K_Q);

    double Mmem[tensor::matM::size()];
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
}

bool Poisson::assemble_volume(std::size_t elNo, Matrix<double>& A00,
                              LinearAllocator<double>& scratch) const {
    double Dx_Q[tensor::Dx_Q::size()];

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

    double K_Dx_q0[tensor::K_Dx_q::size(0)];
    double K_Dx_q1[tensor::K_Dx_q::size(1)];
    auto K_Dx_q = std::array<double*, 2>{K_Dx_q0, K_Dx_q1};
    compute_K_Dx_q(fctNo, info, K_Dx_q);

    double L_q[2][std::max(tensor::L_q::size(0), tensor::L_q::size(1))];

    if (method_ == DGMethod::BR2) {
        double Minv[2][tensor::M::size()];
        for (int i = 0; i < 2; ++i) {
            compute_inverse_mass_matrix(info.up[i], Minv[i]);
        }

        double K_q0[tensor::K_q::size(0)];
        double K_q1[tensor::K_q::size(1)];
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

    double L0[tensor::L_q::size(0)];
    if (method_ == DGMethod::BR2) {
        double Minv0[tensor::M::size()];
        compute_inverse_mass_matrix(info.up[0], Minv0);

        double K_q[tensor::K_q::size(0)];
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

    double K_Dx_q0[tensor::K_Dx_q::size(0)];
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

    double F_Q_raw[tensor::F_Q::size()];
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
    double f_q_raw[tensor::f_q::size()];
    if (!bc_skeleton(fctNo, info.bc, f_q_raw)) {
        return false;
    }

    double f_lifted_q[tensor::f_lifted_q::size()];
    if (method_ == DGMethod::BR2) {
        double Minv[2][tensor::M::size()];
        compute_inverse_mass_matrix(info.up[0], Minv[0]);
        compute_inverse_mass_matrix(info.up[1], Minv[1]);

        double K_q0[tensor::K_q::size(0)];
        double K_q1[tensor::K_q::size(1)];
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

    double K_Dx_q0[tensor::K_Dx_q::size(0)];
    double K_Dx_q1[tensor::K_Dx_q::size(1)];
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
    double f_q_raw[tensor::f_q::size()];
    if (!bc_boundary(fctNo, info.bc, f_q_raw)) {
        return false;
    }

    double f_lifted_q[tensor::f_lifted_q::size()];
    if (method_ == DGMethod::BR2) {
        double M0[tensor::M::size()];
        compute_inverse_mass_matrix(info.up[0], M0);

        double K_q[tensor::K_q::size(0)];
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

    double K_Dx_q0[tensor::K_Dx_q::size(0)];
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

void Poisson::apply(std::size_t elNo, mneme::span<std::size_t> lids,
                    mneme::span<std::size_t> localNos, Vector<double const> const& x_0,
                    std::array<Vector<double const>, NumFacets> const& x_n,
                    Vector<double>& y_0) const {
    double Minv0[tensor::M::size()];
    compute_inverse_mass_matrix(elNo, Minv0);

    double sigma[tensor::sigma::size()];
    kernel::stress_volume sv;
    sv.Dxi_Q = Dxi_Q.data();
    sv.E_Q = E_Q.data();
    sv.G_Q = vol[elNo].get<JInv>().data()->data();
    sv.J_Q = vol[elNo].get<AbsDetJ>().data();
    sv.K = material[elNo].get<K>().data();
    sv.Minv(0) = Minv0;
    sv.U = x_0.data();
    sv.W = volRule.weights().data();
    sv.matDxi_Q = matDxi_Q.data();
    sv.matE_Q_T = matE_Q_T.data();
    sv.sigma = sigma;
    sv.execute();

    double u_hat_q[tensor::u_hat_q::size()];

    for (std::size_t f = 0; f < NumFacets; ++f) {
        kernel::flux_u_skeleton fus;
        fus.E_q(0) = E_q[f].data();
        fus.E_q(1) = E_q[localNos[f]].data();
        fus.U = x_0.data();
        fus.U_ext = x_n[f].data();
        fus.u_hat_q = u_hat_q;
        fus.execute();
    /*generator.add('stress_facet', sigma['ur'] <= sigma['ur'] - Minv[0]['uk'] *
    w['q'] * K['m'] * matE_q_T['qm'] * E_q[0]['kq'] * n_q['rq'] * u_hat_q['q'])

        kernel::stress_facet sf;
        sf.E_q(0) = E_q[f].data();
        sf.K = material[elNo].get<K>().data();
        sf.Minv(0) = Minv0;
        sf.matE_Q_T = matE_Q_T.data();
        sf.n_q = 
        tensor::E_q::Container<double const*> E_q;
        double const* K{};
        tensor::Minv::Container<double const*> Minv;
        double const* matE_q_T{};
        double const* n_q{};
        double* sigma{};
        double const* u_hat_q{};
        double const* w{};*/
        
    }

    /*sigma = Tensor('sigma', (Nbf, dim))
    U = Tensor('U', (Nbf,))
    U_ext = Tensor('U_ext', (Nbf,))
    U_new = Tensor('U_new', (Nbf,))
    u_hat_q = Tensor('u_hat_q', (nq,))
    sigma_hat_q = Tensor('sigma_hat_q', (dim, nq))

    generator.add('flux_u_skeleton',
        u_hat_q['q'] <= 0.5 * (E_q[0]['lq'] * U['l'] + E_q[1]['lq'] * U_ext['l']))
    generator.add('stress_volume', sigma['ur'] <= Minv[0]['uk'] * W['q'] * J_Q['q'] *
        G_Q['erq'] * (K['m'] * E_Q['kq'] * matDxi_Q['meq'] + K['m'] * matE_Q_T['qm'] * Dxi_Q['keq'])
    * E_Q['lq'] * U['l'])
    generator.add('stress_facet', sigma['ur'] <= sigma['ur'] - Minv[0]['uk'] *
    w['q'] * K['m'] * matE_q_T['qm'] * E_q[0]['kq'] * n_q['rq'] * u_hat_q['q'])

    generator.add('flux_sigma_skeleton', sigma_hat_q['pq'] <= 0.5 *
            (K_Dx_q[0]['lpq'] * U['l'] + K_Dx_q[1]['lpq'] * U_ext['l']) +
            c0[0] * (E_q[0]['lq'] * U['l'] - E_q[1]['lq'] * U_ext['l']) * n_unit_q['pq'])
    generator.add('flux_sigma_boundary', sigma_hat_q['pq'] <=
            K_Dx_q[0]['lpq'] * U['l'] + c0[0] * E_q[0]['lq'] * U['l'] * n_unit_q['pq'])
    generator.add('apply_volume', U_new['k'] <= W['q'] * J_Q['q'] * E_Q['lq'] *
        G_Q['erq'] * Dxi_Q['keq'] * sigma['lr'])
    generator.add('apply_facet', U_new['k'] <= U_new['k'] -
        w['q'] * E_q[0]['kq'] * n_q['rq'] * sigma_hat_q['rq'])*/
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

    double f_q_raw[tensor::f_q::size()];
    bc_skeleton(fctNo, info.bc, f_q_raw);

    double K_Dx_q0[tensor::K_Dx_q::size(0)];
    double K_Dx_q1[tensor::K_Dx_q::size(1)];
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

    double f_q_raw[tensor::f_q::size()];
    bc_boundary(fctNo, info.bc, f_q_raw);

    double K_Dx_q0[tensor::K_Dx_q::size(0)];
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
