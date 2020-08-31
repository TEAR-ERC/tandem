#include "Elasticity.h"

#include "basis/WarpAndBlend.h"
#include "config.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "kernels/elasticity/init.h"
#include "kernels/elasticity/kernel.h"
#include "kernels/elasticity/tensor.h"
#include "parallel/MPITraits.h"
#include "tensor/EigenMap.h"

#include <limits>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

namespace tensor = tndm::elasticity::tensor;
namespace init = tndm::elasticity::init;
namespace kernel = tndm::elasticity::kernel;

namespace tndm {

Elasticity::Elasticity(LocalSimplexMesh<DomainDimension> const& mesh,
                       Curvilinear<DomainDimension>& cl,
                       std::unique_ptr<RefElement<DomainDimension>> refElement,
                       unsigned minQuadOrder, MPI_Comm comm, functional_t const& lambdaFun,
                       functional_t const& muFun)
    : DG(mesh, cl, std::move(refElement), minQuadOrder, comm),
      nodalRefElement_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>()),
      facetRefElement_(PolynomialDegree, WarpAndBlendFactory<DomainDimension - 1u>()) {

    int rank;
    MPI_Comm_rank(comm, &rank);

    userVol.setStorage(
        std::make_shared<user_vol_t>(numElements() * nodalRefElement_.numBasisFunctions()), 0u,
        numElements(), nodalRefElement_.numBasisFunctions());

    userVolPre.setStorage(std::make_shared<user_vol_pre_t>(numElements() * volRule.size()), 0u,
                          numElements(), volRule.size());

    userFctPre.setStorage(std::make_shared<user_fct_pre_t>(numLocalFacets() * fctRule.size()), 0u,
                          numLocalFacets(), fctRule.size());

    minv = facetRefElement_.inverseMassMatrix();
    enodal = facetRefElement_.evaluateBasisAt(fctRule.points(), {1, 0});

    auto Ematerial = nodalRefElement_.evaluateBasisAt(volRule.points(), {1, 0});
    std::vector<Managed<Matrix<double>>> ematerial;
    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto points = cl.facetParam(f, fctRule.points());
        ematerial.emplace_back(nodalRefElement_.evaluateBasisAt(points, {1, 0}));
    }

    auto Minv = nodalRefElement_.inverseMassMatrix();
    auto Enodal = nodalRefElement_.evaluateBasisAt(volRule.points());
    Eigen::MatrixXd P = EigenMap(Minv) * EigenMap(Enodal);

#pragma omp parallel
    {
        auto rhs = Eigen::VectorXd(volRule.size());
        // Lua functions not thread-safe
#pragma omp single
        for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
            auto l = Vector<double>(userVol[elNo].get<lam>().data(),
                                    nodalRefElement_.numBasisFunctions());
            auto m = Vector<double>(userVol[elNo].get<mu>().data(),
                                    nodalRefElement_.numBasisFunctions());

            auto coords = vol[elNo].get<Coords>();
            for (std::size_t q = 0; q < volRule.size(); ++q) {
                rhs(q) = lambdaFun(coords[q]) * volRule.weights()[q];
            }
            EigenMap(l) = P * rhs;
            for (std::size_t q = 0; q < volRule.size(); ++q) {
                rhs(q) = muFun(coords[q]) * volRule.weights()[q];
            }
            EigenMap(m) = P * rhs;

            kernel::precomputeVolume krnl;
            krnl.Ematerial = Ematerial.data();
            krnl.J = vol[elNo].template get<AbsDetJ>().data();
            krnl.W = volRule.weights().data();
            krnl.lam = l.data();
            krnl.lam_W_J = userVolPre[elNo].template get<lam_W_J>().data();
            krnl.mu = m.data();
            krnl.mu_W_J = userVolPre[elNo].template get<mu_W_J>().data();
            krnl.execute();
        }

#pragma omp for
        for (std::size_t fctNo = 0; fctNo < numLocalFacets(); ++fctNo) {
            auto const& info = fctInfo[fctNo];
            kernel::precomputeSurface krnl;
            for (unsigned side = 0; side < 2; ++side) {
                krnl.ematerial(side) = ematerial[info.localNo[side]].data();
            }
            krnl.lam_w(0) = userFctPre[fctNo].get<lam_w_0>().data();
            krnl.lam_w(1) = userFctPre[fctNo].get<lam_w_1>().data();
            krnl.mu_w(0) = userFctPre[fctNo].get<mu_w_0>().data();
            krnl.mu_w(1) = userFctPre[fctNo].get<mu_w_1>().data();
            krnl.w = fctRule.weights().data();

            for (unsigned side = 0; side < 2; ++side) {
                krnl.lam = userVol[info.up[side]].get<lam>().data();
                krnl.mu = userVol[info.up[side]].get<mu>().data();
                krnl.execute(side);
            }
        }
    }
}

PetscErrorCode Elasticity::createAShell(Mat* A) const {
    PetscInt blockSize = tensor::U::Shape[0] * tensor::U::Shape[1];
    PetscInt localRows = numLocalElements() * blockSize;
    PetscInt localCols = numLocalElements() * blockSize;
    CHKERRQ(MatCreateShell(comm(), localRows, localCols, PETSC_DETERMINE, PETSC_DETERMINE,
                           const_cast<tndm::Elasticity*>(this), A));
    CHKERRQ(MatShellSetOperation(*A, MATOP_MULT, (void (*)(void))Elasticity::ApplyDGOperator));
    CHKERRQ(MatSetOption(*A, MAT_ROW_ORIENTED, PETSC_FALSE));
    CHKERRQ(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
    return 0;
}

PetscErrorCode Elasticity::assemble(Mat mat) const {
    double A[tensor::A::size()];
    double D_x[tensor::D_x::size()];

    assert(volRule.size() == tensor::W::Shape[0]);
    assert(D_xi.shape(0) == tensor::D_xi::Shape[0]);
    assert(D_xi.shape(1) == tensor::D_xi::Shape[1]);
    assert(D_xi.shape(2) == tensor::D_xi::Shape[2]);

    auto blockSize = tensor::U::Shape[0] * tensor::U::Shape[1];
#pragma omp parallel for private(A, D_x)
    for (std::size_t elNo = 0; elNo < numLocalElements(); ++elNo) {
        kernel::D_x dxKrnl;
        dxKrnl.D_x = D_x;
        dxKrnl.D_xi = D_xi.data();
        dxKrnl.G = vol[elNo].template get<JInv>().data()->data();
        dxKrnl.execute();

        kernel::assembleVolume krnl;
        krnl.A = A;
        krnl.delta = init::delta::Values;
        krnl.D_x = D_x;
        krnl.lam_W_J = userVolPre[elNo].get<lam_W_J>().data();
        krnl.mu_W_J = userVolPre[elNo].get<mu_W_J>().data();
        krnl.execute();

        PetscInt ib = volInfo[elNo].get<GID>();
        PetscInt jb = volInfo[elNo].get<GID>();
        MatSetValuesBlocked(mat, 1, &ib, 1, &jb, A, ADD_VALUES);
    }

    double d_x0[tensor::d_x::size(0)];
    double d_x1[tensor::d_x::size(1)];
    double a00[tensor::a::size(0, 0)];
    double a01[tensor::a::size(0, 1)];
    double a10[tensor::a::size(1, 0)];
    double a11[tensor::a::size(1, 1)];

    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(e[0].shape(0) == tensor::e::Shape[0][0]);
    assert(e[0].shape(1) == tensor::e::Shape[0][1]);
    assert(d_xi[0].shape(0) == tensor::d_xi::Shape[0][0]);
    assert(d_xi[0].shape(1) == tensor::d_xi::Shape[0][1]);
    assert(d_xi[0].shape(2) == tensor::d_xi::Shape[0][2]);

    for (std::size_t fctNo = 0; fctNo < numLocalFacets(); ++fctNo) {
        auto const& info = fctInfo[fctNo];
        if (info.bc == BC::Natural) {
            continue;
        }

        auto push = [&info, &mat](auto x, auto y, double* a) {
            PetscInt ib = info.g_up[x()];
            PetscInt jb = info.g_up[y()];
            MatSetValuesBlocked(mat, 1, &ib, 1, &jb, a, ADD_VALUES);
        };

        kernel::d_x dxKrnl;
        dxKrnl.d_x(0) = d_x0;
        dxKrnl.d_x(1) = d_x1;
        dxKrnl.d_xi(0) = d_xi[info.localNo[0]].data();
        dxKrnl.d_xi(1) = d_xi[info.localNo[1]].data();
        dxKrnl.g(0) = fct[fctNo].template get<JInv>().data()->data();
        dxKrnl.g(1) = fct[fctNo].template get<JInvOther>().data()->data();

        double half = (info.up[0] != info.up[1]) ? 0.5 : 1.0;
        kernel::assembleSurface krnl;
        krnl.c00 = -half;
        krnl.c01 = -krnl.c00;
        krnl.c10 = epsilon * half;
        krnl.c11 = -krnl.c10;
        krnl.c20 = penalty(info);
        krnl.c21 = -krnl.c20;
        krnl.d_x(0) = d_x0;
        krnl.d_x(1) = d_x1;
        for (unsigned side = 0; side < 2; ++side) {
            krnl.e(side) = e[info.localNo[side]].data();
        }
        krnl.lam_w(0) = userFctPre[fctNo].template get<lam_w_0>().data();
        krnl.lam_w(1) = userFctPre[fctNo].template get<lam_w_1>().data();
        krnl.mu_w(0) = userFctPre[fctNo].template get<mu_w_0>().data();
        krnl.mu_w(1) = userFctPre[fctNo].template get<mu_w_1>().data();
        krnl.n = fct[fctNo].template get<Normal>().data()->data();
        krnl.nl = fct[fctNo].template get<NormalLength>().data();
        krnl.w = fctRule.weights().data();
        krnl.a(0, 0) = a00;
        krnl.a(0, 1) = a01;
        krnl.a(1, 0) = a10;
        krnl.a(1, 1) = a11;
        krnl.delta = init::delta::Values;

        dxKrnl.execute(0);
        if (info.inside[0]) {
            krnl.execute(0, 0);
            push(std::integral_constant<int, 0>(), std::integral_constant<int, 0>(), a00);
        }

        if (info.up[0] != info.up[1]) {
            dxKrnl.execute(1);
            if (info.inside[0]) {
                krnl.execute(0, 1);
                push(std::integral_constant<int, 0>(), std::integral_constant<int, 1>(), a01);
            }
            if (info.inside[1]) {
                krnl.execute(1, 0);
                push(std::integral_constant<int, 1>(), std::integral_constant<int, 0>(), a10);
                krnl.execute(1, 1);
                push(std::integral_constant<int, 1>(), std::integral_constant<int, 1>(), a11);
            }
        }
    }

    CHKERRQ(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
    return 0;
}

void Elasticity::apply(double const* U, double* Unew) const {
    double D_x[tensor::D_x::size()];

    assert(volRule.size() == tensor::W::Shape[0]);
    assert(D_xi.shape(0) == tensor::D_xi::Shape[0]);
    assert(D_xi.shape(1) == tensor::D_xi::Shape[1]);
    assert(D_xi.shape(2) == tensor::D_xi::Shape[2]);

    auto blockSize = tensor::U::Shape[0] * tensor::U::Shape[1];
#pragma omp parallel for private(D_x)
    for (std::size_t elNo = 0; elNo < numLocalElements(); ++elNo) {
        kernel::D_x dxKrnl;
        dxKrnl.D_x = D_x;
        dxKrnl.D_xi = D_xi.data();
        dxKrnl.G = vol[elNo].template get<JInv>().data()->data();
        dxKrnl.execute();

        kernel::volumeOp krnl;
        krnl.D_x = D_x;
        krnl.U = U + elNo * blockSize;
        krnl.Unew = Unew + elNo * blockSize;
        krnl.lam_W_J = userVolPre[elNo].get<lam_W_J>().data();
        krnl.mu_W_J = userVolPre[elNo].get<mu_W_J>().data();
        krnl.execute();
    }

    double d_x0[tensor::d_x::size(0)];
    double d_x1[tensor::d_x::size(1)];
    double traction_avg[tensor::traction_avg::size()];
    double u_jump[tensor::u_jump::size()];

    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(e[0].shape(0) == tensor::e::Shape[0][0]);
    assert(e[0].shape(1) == tensor::e::Shape[0][1]);
    assert(d_xi[0].shape(0) == tensor::d_xi::Shape[0][0]);
    assert(d_xi[0].shape(1) == tensor::d_xi::Shape[0][1]);
    assert(d_xi[0].shape(2) == tensor::d_xi::Shape[0][2]);

    //#pragma omp parallel for private(d_x0, d_x1, u_jump, traction_avg)
    for (std::size_t fctNo = 0; fctNo < numLocalFacets(); ++fctNo) {
        auto const& info = fctInfo[fctNo];
        if (info.bc == BC::Natural) {
            continue;
        }

        kernel::d_x dxKrnl;
        dxKrnl.d_x(0) = d_x0;
        dxKrnl.d_x(1) = d_x1;
        dxKrnl.d_xi(0) = d_xi[info.localNo[0]].data();
        dxKrnl.d_xi(1) = d_xi[info.localNo[1]].data();
        dxKrnl.g(0) = fct[fctNo].template get<JInv>().data()->data();
        dxKrnl.g(1) = fct[fctNo].template get<JInvOther>().data()->data();
        dxKrnl.execute(0);

        if (info.up[0] == info.up[1]) {
            kernel::surfaceOpBnd krnl;
            krnl.c00 = -1.0;
            krnl.c10 = epsilon;
            krnl.c20 = penalty(info);
            krnl.d_x(0) = d_x0;
            krnl.e(0) = e[info.localNo[0]].data();
            krnl.u(0) = U + info.up[0] * blockSize;
            krnl.unew(0) = Unew + info.up[0] * blockSize;
            krnl.lam_w(0) = userFctPre[fctNo].template get<lam_w_0>().data();
            krnl.mu_w(0) = userFctPre[fctNo].template get<mu_w_0>().data();
            krnl.n = fct[fctNo].template get<Normal>().data()->data();
            krnl.nl = fct[fctNo].template get<NormalLength>().data();
            krnl.w = fctRule.weights().data();
            krnl.traction_avg = traction_avg;
            krnl.u_jump = u_jump;
            krnl.execute();
        } else {
            dxKrnl.execute(1);

            kernel::surfaceOp krnl;
            krnl.c00 = -1.0;
            krnl.c01 = -krnl.c00;
            krnl.c10 = epsilon * 0.5;
            krnl.c11 = krnl.c10;
            krnl.c20 = penalty(info);
            krnl.c21 = -krnl.c20;
            krnl.d_x(0) = d_x0;
            krnl.d_x(1) = d_x1;
            for (unsigned side = 0; side < 2; ++side) {
                krnl.e(side) = e[info.localNo[side]].data();
                krnl.u(side) = U + info.up[side] * blockSize;
                krnl.unew(side) = Unew + info.up[side] * blockSize;
            }
            krnl.lam_w(0) = userFctPre[fctNo].template get<lam_w_0>().data();
            krnl.lam_w(1) = userFctPre[fctNo].template get<lam_w_1>().data();
            krnl.mu_w(0) = userFctPre[fctNo].template get<mu_w_0>().data();
            krnl.mu_w(1) = userFctPre[fctNo].template get<mu_w_1>().data();
            krnl.n = fct[fctNo].template get<Normal>().data()->data();
            krnl.nl = fct[fctNo].template get<NormalLength>().data();
            krnl.w = fctRule.weights().data();
            krnl.traction_avg = traction_avg;
            krnl.u_jump = u_jump;
            krnl.execute();
        }
    }
}

PetscErrorCode Elasticity::rhs(Vec B, volume_functional_t forceFun, facet_functional_t dirichletFun,
                               facet_functional_t slipFun) const {
    PetscErrorCode ierr;

    double b[tensor::b::size()];
    double F[tensor::F::size()];
    for (std::size_t elNo = 0; elNo < numLocalElements(); ++elNo) {
        forceFun(elNo, F);

        kernel::rhsVolume rhs;
        rhs.E = E.data();
        rhs.F = F;
        rhs.J = vol[elNo].template get<AbsDetJ>().data();
        rhs.W = volRule.weights().data();
        rhs.b = b;
        rhs.execute();

        PetscInt ib = volInfo[elNo].get<GID>();
        VecSetValuesBlocked(B, 1, &ib, b, ADD_VALUES);
    }

    double f[tensor::f::size()];
    double d_x[tensor::d_x::size(0)];

    for (std::size_t fctNo = 0; fctNo < numLocalFacets(); ++fctNo) {
        auto const& info = fctInfo[fctNo];
        if (info.bc == BC::Fault || info.bc == BC::Dirichlet) {
            facet_functional_t& fun = (info.bc == BC::Fault) ? slipFun : dirichletFun;
            fun(fctNo, f);

            double half = (info.up[0] != info.up[1]) ? 0.5 : 1.0;
            kernel::rhsFacet rhs;
            rhs.b = b;
            rhs.c10 = half * epsilon;
            rhs.c20 = penalty(info);
            rhs.d_x(0) = d_x;
            rhs.f = f;
            rhs.n = fct[fctNo].template get<Normal>().data()->data();
            rhs.nl = fct[fctNo].template get<NormalLength>().data();
            rhs.w = fctRule.weights().data();
            if (info.inside[0]) {
                rhs.d_xi(0) = d_xi[info.localNo[0]].data();
                rhs.e(0) = e[info.localNo[0]].data();
                rhs.g(0) = fct[fctNo].template get<JInv>().data()->data();
                rhs.lam_w(0) = userFctPre[fctNo].template get<lam_w_0>().data();
                rhs.mu_w(0) = userFctPre[fctNo].template get<mu_w_0>().data();
                rhs.execute();

                PetscInt ib = info.g_up[0];
                VecSetValuesBlocked(B, 1, &ib, b, ADD_VALUES);
            }
            if (info.inside[1] && info.up[0] != info.up[1]) {
                rhs.c20 *= -1.0;
                rhs.d_xi(0) = d_xi[info.localNo[1]].data();
                rhs.e(0) = e[info.localNo[1]].data();
                rhs.g(0) = fct[fctNo].template get<JInvOther>().data()->data();
                rhs.lam_w(0) = userFctPre[fctNo].template get<lam_w_1>().data();
                rhs.mu_w(0) = userFctPre[fctNo].template get<mu_w_1>().data();
                rhs.execute();

                PetscInt ib = info.g_up[1];
                VecSetValuesBlocked(B, 1, &ib, b, ADD_VALUES);
            }
        }
    }

    CHKERRQ(VecAssemblyBegin(B));
    CHKERRQ(VecAssemblyEnd(B));
    return 0;
}

auto Elasticity::makeVolumeFunctional(vector_functional_t fun) const -> volume_functional_t {
    return [fun, this](std::size_t elNo, double* F) {
        auto Fview = init::F::view::create(F);
        auto coords = this->vol[elNo].template get<Coords>();
        for (unsigned q = 0; q < tensor::F::Shape[1]; ++q) {
            auto fx = fun(coords[q]);
            for (unsigned p = 0; p < tensor::F::Shape[0]; ++p) {
                Fview(p, q) = fx[p];
            }
        }
    };
}

auto Elasticity::makeFacetFunctional(vector_functional_t fun) const -> facet_functional_t {
    return [fun, this](std::size_t fctNo, double* f) {
        auto coords = this->fct[fctNo].template get<Coords>();
        auto const& info = this->fctInfo[fctNo];
        auto fview = init::f::view::create(f);
        for (unsigned q = 0; q < tensor::f::Shape[1]; ++q) {
            auto fx = fun(coords[q]);
            for (unsigned p = 0; p < tensor::f::Shape[0]; ++p) {
                fview(p, q) = fx[p];
            }
            // \todo Make side detection configurable
            auto n = this->fct[fctNo].template get<Normal>()[q];
            if (info.bc == BC::Fault && n[0] < 0.0) {
                for (unsigned p = 0; p < tensor::f::Shape[0]; ++p) {
                    fview(p, q) *= -1.0;
                }
            }
        }
    };
}

FiniteElementFunction<DomainDimension> Elasticity::finiteElementFunction(Vec x) const {
    auto numeric = FiniteElementFunction<DomainDimension>(refElement_->clone(), tensor::U::Shape[0],
                                                          tensor::U::Shape[1], numLocalElements());
    PetscScalar const* values;
    VecGetArrayRead(x, &values);
    auto& data = numeric.values();
    auto blockSize = tensor::U::Shape[0] * tensor::U::Shape[1];
    for (std::size_t elNo = 0; elNo < numLocalElements(); ++elNo) {
        for (std::size_t p = 0; p < tensor::U::Shape[1]; ++p) {
            for (std::size_t dof = 0; dof < tensor::U::Shape[0]; ++dof) {
                data(dof, p, elNo) = values[dof + p * tensor::U::Shape[0] + elNo * blockSize];
            }
        }
    }
    VecRestoreArrayRead(x, &values);
    return numeric;
}

void Elasticity::traction(std::size_t fctNo, double const* Rnodal, double const* U,
                          Matrix<double>& result) const {
    assert(result.size() == tensor::traction_avg_proj::size());

    double d_x0[tensor::d_x::size(0)];
    double d_x1[tensor::d_x::size(1)];
    double traction_avg[tensor::traction_avg::size()];

    auto blockSize = tensor::U::Shape[0] * tensor::U::Shape[1];
    auto const& info = fctInfo[fctNo];
    kernel::d_x dxKrnl;
    dxKrnl.d_x(0) = d_x0;
    dxKrnl.d_x(1) = d_x1;
    dxKrnl.d_xi(0) = d_xi[info.localNo[0]].data();
    dxKrnl.d_xi(1) = d_xi[info.localNo[1]].data();
    dxKrnl.g(0) = fct[fctNo].template get<JInv>().data()->data();
    dxKrnl.g(1) = fct[fctNo].template get<JInvOther>().data()->data();
    dxKrnl.execute(0);
    dxKrnl.execute(1);

    kernel::traction_avg_proj krnl;
    krnl.d_x(0) = d_x0;
    krnl.d_x(1) = d_x1;
    krnl.enodal = enodal.data();
    krnl.lam_w(0) = userFctPre[fctNo].template get<lam_w_0>().data();
    krnl.lam_w(1) = userFctPre[fctNo].template get<lam_w_1>().data();
    krnl.minv = minv.data();
    krnl.mu_w(0) = userFctPre[fctNo].template get<mu_w_0>().data();
    krnl.mu_w(1) = userFctPre[fctNo].template get<mu_w_1>().data();
    krnl.n = fct[fctNo].template get<Normal>().data()->data();
    krnl.Rnodal = Rnodal;
    krnl.traction_avg = traction_avg;
    krnl.traction_avg_proj = result.data();
    krnl.u(0) = U + info.up[0] * blockSize;
    krnl.u(1) = U + info.up[1] * blockSize;
    krnl.execute();
}

} // namespace tndm
