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

#include <cassert>

namespace tensor = tndm::poisson::tensor;
namespace init = tndm::poisson::init;
namespace kernel = tndm::poisson::kernel;

namespace tndm::tmp {

Poisson::Poisson(Curvilinear<DomainDimension> const& cl, functional_t<1> K)
    : DGCurvilinearCommon<DomainDimension>(cl, MinQuadOrder()), space_(PolynomialDegree),
      materialSpace_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>()),
      fun_K(make_volume_functional(std::move(K))), fun_force(zero_function),
      fun_dirichlet(zero_function), fun_slip(zero_function) {

    E_Q = space_.evaluateBasisAt(volRule.points());
    Dxi_Q = space_.evaluateGradientAt(volRule.points());
    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto points = cl.facetParam(f, fctRule.points());
        E_q.emplace_back(space_.evaluateBasisAt(points));
        Dxi_q.emplace_back(space_.evaluateGradientAt(points));
        matE_q_T.emplace_back(materialSpace_.evaluateBasisAt(points, {1, 0}));
    }

    matMinv = materialSpace_.inverseMassMatrix();
    matE_Q_T = materialSpace_.evaluateBasisAt(volRule.points(), {1, 0});
}

void Poisson::begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                                std::size_t numLocalFacets) {
    base::begin_preparation(numElements, numLocalElements, numLocalFacets);

    material.setStorage(
        std::make_shared<material_vol_t>(numElements * materialSpace_.numBasisFunctions()), 0u,
        numElements, materialSpace_.numBasisFunctions());
}

void Poisson::prepare_volume(std::size_t elNo, LinearAllocator& scratch) {
    base::prepare_volume(elNo, scratch);

    auto Kfield = material[elNo].get<K>().data();
    double K_Q_raw[tensor::K_Q::size()];
    auto K_Q = Matrix<double>(K_Q_raw, 1, volRule.size());
    fun_K(elNo, K_Q);

    kernel::project_K krnl;
    krnl.Em = matE_Q_T.data();
    krnl.K = Kfield;
    krnl.K_Q = K_Q_raw;
    krnl.W = volRule.weights().data();
    krnl.matMinv = matMinv.data();
    krnl.execute();

    auto Kmax = *std::max_element(Kfield, Kfield + materialSpace_.numBasisFunctions());
    base::penalty[elNo] *=
        Kmax * (PolynomialDegree + 1) * (PolynomialDegree + DomainDimension) / DomainDimension;
}

bool Poisson::assemble_volume(std::size_t elNo, Matrix<double>& A00,
                              LinearAllocator& scratch) const {
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
                                LinearAllocator& scratch) const {
    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(E_q[0].shape(0) == tensor::e::Shape[0][0]);
    assert(E_q[0].shape(1) == tensor::e::Shape[0][1]);
    assert(Dxi_q[0].shape(0) == tensor::d_xi::Shape[0][0]);
    assert(Dxi_q[0].shape(1) == tensor::d_xi::Shape[0][1]);
    assert(Dxi_q[0].shape(2) == tensor::d_xi::Shape[0][2]);

    double Dx_q0[tensor::d_x::size(0)];
    double Dx_q1[tensor::d_x::size(1)];

    kernel::assembleFacetLocal local;
    local.c00 = -0.5;
    local.c10 = epsilon * 0.5;
    local.c20 = penalty(info);
    local.a(0, 0) = A00.data();
    local.d_x(0) = Dx_q0;
    local.d_xi(0) = Dxi_q[info.localNo[0]].data();
    local.e(0) = E_q[info.localNo[0]].data();
    local.em(0) = matE_q_T[info.localNo[0]].data();
    local.g(0) = fct[fctNo].get<JInv0>().data()->data();
    local.K = material[info.up[0]].get<K>().data();
    local.n = fct[fctNo].get<Normal>().data()->data();
    local.nl = fct[fctNo].get<NormalLength>().data();
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
    neighbour.n = local.n;
    neighbour.nl = local.nl;
    neighbour.w = local.w;
    neighbour.execute();
    return true;
}

bool Poisson::assemble_boundary(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                                Matrix<double>& A01, LinearAllocator& scratch) const {
    if (info.bc == BC::Natural) {
        return false;
    }

    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(E_q[0].shape(0) == tensor::e::Shape[0][0]);
    assert(E_q[0].shape(1) == tensor::e::Shape[0][1]);
    assert(Dxi_q[0].shape(0) == tensor::d_xi::Shape[0][0]);
    assert(Dxi_q[0].shape(1) == tensor::d_xi::Shape[0][1]);
    assert(Dxi_q[0].shape(2) == tensor::d_xi::Shape[0][2]);

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
    local.n = fct[fctNo].get<Normal>().data()->data();
    local.nl = fct[fctNo].get<NormalLength>().data();
    local.w = fctRule.weights().data();
    local.execute();
    return true;
}

bool Poisson::rhs_volume(std::size_t elNo, Vector<double>& B, LinearAllocator& scratch) const {
    assert(tensor::b::Shape[0] == tensor::A::Shape[0]);

    double F_Q_raw[tensor::F::size()];
    assert(tensor::F::size() == volRule.size());

    auto F_Q = Matrix<double>(F_Q_raw, 1, volRule.size());
    fun_force(elNo, F_Q);

    kernel::rhsVolume rhs;
    rhs.E = E_Q.data();
    rhs.F = F_Q_raw;
    rhs.J = vol[elNo].get<AbsDetJ>().data();
    rhs.W = volRule.weights().data();
    rhs.b = B.data();
    rhs.execute();
    return true;
}

bool Poisson::rhs_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                           Vector<double>& B1, LinearAllocator& scratch) const {
    if (info.bc != BC::Fault) {
        return false;
    }

    double f_q_raw[tensor::f::size()];
    assert(tensor::f::size() == fctRule.size());

    auto f_q = Matrix<double>(f_q_raw, 1, fctRule.size());
    fun_slip(fctNo, f_q);

    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = 0.5 * epsilon;
    rhs.c20 = penalty(info);
    rhs.f = f_q_raw;
    rhs.n = fct[fctNo].get<Normal>().data()->data();
    rhs.nl = fct[fctNo].get<NormalLength>().data();
    rhs.w = fctRule.weights().data();
    rhs.d_xi(0) = Dxi_q[info.localNo[0]].data();
    rhs.e(0) = E_q[info.localNo[0]].data();
    rhs.em(0) = matE_q_T[info.localNo[0]].data();
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.K = material[info.up[0]].get<K>().data();
    rhs.execute();

    rhs.b = B1.data();
    rhs.c20 *= -1.0;
    rhs.d_xi(0) = Dxi_q[info.localNo[1]].data();
    rhs.e(0) = E_q[info.localNo[1]].data();
    rhs.em(0) = matE_q_T[info.localNo[1]].data();
    rhs.g(0) = fct[fctNo].get<JInv1>().data()->data();
    rhs.K = material[info.up[1]].get<K>().data();
    rhs.execute();
    return true;
}

bool Poisson::rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                           LinearAllocator& scratch) const {
    if (info.bc != BC::Dirichlet) {
        return false;
    }

    double f_q_raw[tensor::f::size()];
    assert(tensor::f::size() == fctRule.size());

    auto f_q = Matrix<double>(f_q_raw, 1, fctRule.size());
    fun_dirichlet(fctNo, f_q);

    kernel::rhsFacet rhs;
    rhs.b = B0.data();
    rhs.c10 = epsilon;
    rhs.c20 = penalty(info);
    rhs.f = f_q_raw;
    rhs.n = fct[fctNo].get<Normal>().data()->data();
    rhs.nl = fct[fctNo].get<NormalLength>().data();
    rhs.w = fctRule.weights().data();
    rhs.d_xi(0) = Dxi_q[info.localNo[0]].data();
    rhs.e(0) = E_q[info.localNo[0]].data();
    rhs.em(0) = matE_q_T[info.localNo[0]].data();
    rhs.g(0) = fct[fctNo].get<JInv0>().data()->data();
    rhs.K = material[info.up[0]].get<K>().data();
    rhs.execute();
    return true;
}

void Poisson::coefficients_volume(std::size_t elNo, Matrix<double>& C, LinearAllocator&) const {
    auto const coeff_K = material[elNo].get<K>();
    assert(coeff_K.size() == C.shape(0));
    for (std::size_t i = 0; i < coeff_K.size(); ++i) {
        C(i, 0) = coeff_K[i];
    }
}

} // namespace tndm::tmp
