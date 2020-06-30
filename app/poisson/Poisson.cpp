#include "Poisson.h"

#include "config.h"
#include "form/FiniteElementFunction.h"
#include "kernels/init.h"
#include "kernels/kernel.h"
#include "kernels/tensor.h"

namespace tndm {

Eigen::SparseMatrix<double> Poisson::assemble() {
    using T = Eigen::Triplet<double>;
    std::vector<T> triplets;

    double D_x[tensor::D_x::size()];
    double A[tensor::A::size()];
    auto Aview = init::A::view::create(A);

    assert(volRule.size() == tensor::W::Shape[0]);
    assert(D_xi.shape(0) == tensor::D_xi::Shape[0]);
    assert(D_xi.shape(1) == tensor::D_xi::Shape[1]);
    assert(D_xi.shape(2) == tensor::D_xi::Shape[2]);

    for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
        kernel::assembleVolume krnl;
        krnl.A = A;
        krnl.D_x = D_x;
        krnl.D_xi = D_xi.data();
        krnl.G = vol[elNo].template get<JInv>().data()->data();
        krnl.J = vol[elNo].template get<AbsDetJ>().data();
        krnl.W = volRule.weights().data();
        krnl.execute();
        unsigned i0 = elNo * tensor::A::Shape[0];
        unsigned j0 = elNo * tensor::A::Shape[1];
        for (unsigned i = 0; i < tensor::A::Shape[0]; ++i) {
            for (unsigned j = 0; j < tensor::A::Shape[1]; ++j) {
                triplets.emplace_back(T(i0 + i, j0 + j, Aview(i, j)));
            }
        }
    }

    double a00[tensor::a::size(0, 0)];
    double a01[tensor::a::size(0, 1)];
    double a10[tensor::a::size(1, 0)];
    double a11[tensor::a::size(1, 1)];
    double d_x0[tensor::d_x::size(0)];
    double d_x1[tensor::d_x::size(1)];

    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(e[0].shape(0) == tensor::e::Shape[0][0]);
    assert(e[0].shape(1) == tensor::e::Shape[0][1]);
    assert(d_xi[0].shape(0) == tensor::d_xi::Shape[0][0]);
    assert(d_xi[0].shape(1) == tensor::d_xi::Shape[0][1]);
    assert(d_xi[0].shape(2) == tensor::d_xi::Shape[0][2]);

    for (std::size_t fctNo = 0; fctNo < numFacets(); ++fctNo) {
        auto const& info = fctInfo[fctNo];
        kernel::assembleFacetLocal local;
        double half = (info.up[0] != info.up[1]) ? 0.5 : 1.0;
        local.c00 = -half;
        local.c10 = epsilon * half;
        local.c20 = penalty / std::pow(info.area, beta0);
        local.a(0, 0) = a00;
        local.d_x(0) = d_x0;
        local.d_xi(0) = d_xi[info.localNo[0]].data();
        local.e(0) = e[info.localNo[0]].data();
        local.g = fct[fctNo].template get<JInv>().data()->data();
        local.n = fct[fctNo].template get<Normal>().data()->data();
        local.nl = fct[fctNo].template get<NormalLength>().data();
        local.w = fctRule.weights().data();
        local.execute();

        auto push = [&info, &triplets](auto x, auto y, double* a) {
            auto aview = init::a::view<x(), y()>::create(a);
            unsigned i0 = info.up[x()] * aview.shape(0);
            unsigned j0 = info.up[y()] * aview.shape(1);
            for (unsigned i = 0; i < aview.shape(0); ++i) {
                for (unsigned j = 0; j < aview.shape(1); ++j) {
                    triplets.emplace_back(T(i0 + i, j0 + j, aview(i, j)));
                }
            }
        };

        push(std::integral_constant<int, 0>(), std::integral_constant<int, 0>(), a00);

        if (info.up[0] != info.up[1]) {
            kernel::assembleFacetNeighbour neighbour;
            neighbour.c00 = local.c00;
            neighbour.c01 = -local.c00;
            neighbour.c10 = local.c10;
            neighbour.c11 = -local.c10;
            neighbour.c20 = local.c20;
            neighbour.c21 = -local.c20;
            neighbour.a(0, 1) = a01;
            neighbour.a(1, 0) = a10;
            neighbour.a(1, 1) = a11;
            neighbour.d_x(0) = d_x0;
            neighbour.d_x(1) = d_x1;
            neighbour.d_xi(1) = d_xi[info.localNo[1]].data();
            neighbour.e(0) = local.e(0);
            neighbour.e(1) = e[info.localNo[1]].data();
            neighbour.g = fct[fctNo].template get<JInvOther>().data()->data();
            neighbour.n = local.n;
            neighbour.nl = local.nl;
            neighbour.w = local.w;
            neighbour.execute();

            push(std::integral_constant<int, 0>(), std::integral_constant<int, 1>(), a01);
            push(std::integral_constant<int, 1>(), std::integral_constant<int, 0>(), a10);
            push(std::integral_constant<int, 1>(), std::integral_constant<int, 1>(), a11);
        }
    }

    Eigen::SparseMatrix<double> mat(numElements() * tensor::A::Shape[0],
                                    numElements() * tensor::A::Shape[1]);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
}

Eigen::VectorXd Poisson::rhs(functional_t forceFun, functional_t dirichletFun) {
    Eigen::VectorXd B = Eigen::VectorXd::Zero(numElements() * tensor::A::Shape[0]);

    double b[tensor::b::size()];
    auto bview = init::b::view::create(b);

    assert(tensor::b::Shape[0] == tensor::A::Shape[0]);

    double F[tensor::F::size()];
    assert(tensor::F::size() == volRule.size());
    for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
        auto coords = vol[elNo].template get<Coords>();
        for (unsigned q = 0; q < tensor::F::size(); ++q) {
            F[q] = forceFun(coords[q]);
        }

        kernel::rhsVolume rhs;
        rhs.E = E.data();
        rhs.F = F;
        rhs.J = vol[elNo].template get<AbsDetJ>().data();
        rhs.W = volRule.weights().data();
        rhs.b = b;
        rhs.execute();

        unsigned i0 = elNo * bview.shape(0);
        for (unsigned i = 0; i < bview.shape(0); ++i) {
            B[i0 + i] += bview(i);
        }
    }

    double f[tensor::f::size()];
    assert(tensor::f::size() == fctRule.size());

    for (std::size_t fctNo = 0; fctNo < numFacets(); ++fctNo) {
        auto const& info = fctInfo[fctNo];
        if (info.up[0] == info.up[1]) {
            auto coords = fct[fctNo].template get<Coords>();
            for (unsigned q = 0; q < tensor::f::size(); ++q) {
                f[q] = dirichletFun(coords[q]);
            }

            kernel::rhsFacet rhs;
            rhs.c10 = epsilon;
            rhs.c20 = penalty / std::pow(info.area, beta0);
            rhs.b = b;
            rhs.d_xi(0) = d_xi[info.localNo[0]].data();
            rhs.e(0) = e[info.localNo[0]].data();
            rhs.f = f;
            rhs.g = fct[fctNo].template get<JInv>().data()->data();
            rhs.n = fct[fctNo].template get<Normal>().data()->data();
            rhs.nl = fct[fctNo].template get<NormalLength>().data();
            rhs.w = fctRule.weights().data();
            rhs.execute();

            unsigned i0 = info.up[0] * bview.shape(0);
            for (unsigned i = 0; i < bview.shape(0); ++i) {
                B[i0 + i] += bview(i);
            }
        }
    }
    return B;
}

FiniteElementFunction<DomainDimension>
Poisson::finiteElementFunction(Eigen::VectorXd const& numeric) const {
    assert(numeric.rows() == numElements() * tensor::b::Shape[0]);
    return FiniteElementFunction<DomainDimension>(refElement_->clone(), numeric.data(),
                                                  tensor::b::Shape[0], 1, numElements());
}

} // namespace tndm
