#include "Elasticity.h"

#include "basis/WarpAndBlend.h"
#include "config.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "kernels/tandem/init.h"
#include "kernels/tandem/kernel.h"
#include "kernels/tandem/tensor.h"
#include "parallel/MPITraits.h"
#include "tensor/EigenMap.h"

#include <limits>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

namespace tndm {

Elasticity::Elasticity(LocalSimplexMesh<DomainDimension> const& mesh,
                       Curvilinear<DomainDimension>& cl,
                       std::unique_ptr<RefElement<DomainDimension>> refElement,
                       unsigned minQuadOrder, MPI_Comm comm, functional_t const& lambdaFun,
                       functional_t const& muFun)
    : DG(mesh, cl, std::move(refElement), minQuadOrder, comm),
      nodalRefElement_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>()) {

    int rank;
    MPI_Comm_rank(comm, &rank);

    userVol.setStorage(
        std::make_shared<user_vol_t>(numElements() * nodalRefElement_.numBasisFunctions()), 0u,
        numElements(), nodalRefElement_.numBasisFunctions());

    userVolPre.setStorage(std::make_shared<user_vol_pre_t>(numElements() * volRule.size()), 0u,
                          numElements(), volRule.size());

    userFctPre.setStorage(std::make_shared<user_fct_pre_t>(numLocalFacets() * fctRule.size()), 0u,
                          numLocalFacets(), fctRule.size());

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
#pragma omp for
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

PetscErrorCode Elasticity::createA(Mat* A) {
    PetscInt blockSize = tensor::U::Shape[0] * tensor::U::Shape[1];
    PetscInt localRows = numLocalElements() * blockSize;
    PetscInt localCols = numLocalElements() * blockSize;
    CHKERRQ(
        MatCreateShell(comm(), localRows, localCols, PETSC_DETERMINE, PETSC_DETERMINE, this, A));
    CHKERRQ(MatShellSetOperation(*A, MATOP_MULT, (void (*)(void))Elasticity::ApplyDGOperator));
    CHKERRQ(MatSetOption(*A, MAT_ROW_ORIENTED, PETSC_FALSE));
    CHKERRQ(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
    return 0;
}
PetscErrorCode Elasticity::createb(Vec* b) {
    PetscInt blockSize = tensor::U::Shape[0] * tensor::U::Shape[1];
    PetscInt localRows = numLocalElements() * blockSize;
    CHKERRQ(VecCreate(comm(), b));
    CHKERRQ(VecSetSizes(*b, localRows, PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(*b));
    CHKERRQ(VecSetBlockSize(*b, blockSize));
    return 0;
}

void Elasticity::apply(double const* U, double* Unew) {
    double D_x[tensor::D_x::size()];

    assert(volRule.size() == tensor::W::Shape[0]);
    assert(D_xi.shape(0) == tensor::D_xi::Shape[0]);
    assert(D_xi.shape(1) == tensor::D_xi::Shape[1]);
    assert(D_xi.shape(2) == tensor::D_xi::Shape[2]);

    auto blockSize = tensor::U::Shape[0] * tensor::U::Shape[1];
#pragma omp parallel for private(D_x)
    for (std::size_t elNo = 0; elNo < numLocalElements(); ++elNo) {
        kernel::volumeOp krnl;
        krnl.D_x = D_x;
        krnl.D_xi = D_xi.data();
        krnl.G = vol[elNo].template get<JInv>().data()->data();
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

#pragma omp parallel for private(d_x0, d_x1, u_jump, traction_avg)
    for (std::size_t fctNo = 0; fctNo < numLocalFacets(); ++fctNo) {
        auto const& info = fctInfo[fctNo];
        if (info.bc == BC::Fault || info.bc == BC::Natural) {
            continue;
        }

        if (info.up[0] == info.up[1]) {
            kernel::surfaceOpBnd krnl;
            krnl.c00 = -1.0;
            krnl.c10 = epsilon;
            krnl.c20 = penalty(info);
            krnl.d_x(0) = d_x0;
            krnl.d_xi(0) = d_xi[info.localNo[0]].data();
            krnl.e(0) = e[info.localNo[0]].data();
            krnl.u(0) = U + info.up[0] * blockSize;
            krnl.unew(0) = Unew + info.up[0] * blockSize;
            krnl.g(0) = fct[fctNo].template get<JInv>().data()->data();
            krnl.lam_w(0) = userFctPre[fctNo].template get<lam_w_0>().data();
            krnl.mu_w(0) = userFctPre[fctNo].template get<mu_w_0>().data();
            krnl.n = fct[fctNo].template get<Normal>().data()->data();
            krnl.nl = fct[fctNo].template get<NormalLength>().data();
            krnl.w = fctRule.weights().data();
            krnl.traction_avg = traction_avg;
            krnl.u_jump = u_jump;
            krnl.execute();
        } else {
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
                krnl.d_xi(side) = d_xi[info.localNo[side]].data();
                krnl.e(side) = e[info.localNo[side]].data();
                krnl.u(side) = U + info.up[side] * blockSize;
                krnl.unew(side) = Unew + info.up[side] * blockSize;
            }
            krnl.g(0) = fct[fctNo].template get<JInv>().data()->data();
            krnl.g(1) = fct[fctNo].template get<JInvOther>().data()->data();
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

PetscErrorCode Elasticity::rhs(Vec B, vector_functional_t forceFun,
                               vector_functional_t dirichletFun) {
    PetscErrorCode ierr;

    double b[tensor::b::size()];
    double F[tensor::F::size()];
    auto Fview = init::F::view::create(F);
    for (std::size_t elNo = 0; elNo < numLocalElements(); ++elNo) {
        auto coords = vol[elNo].template get<Coords>();
        for (unsigned q = 0; q < tensor::F::Shape[1]; ++q) {
            auto f = forceFun(coords[q]);
            for (unsigned p = 0; p < tensor::F::Shape[0]; ++p) {
                Fview(p, q) = f[p];
            }
        }

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
    auto fview = init::f::view::create(f);

    for (std::size_t fctNo = 0; fctNo < numLocalFacets(); ++fctNo) {
        auto const& info = fctInfo[fctNo];
        if (info.up[0] == info.up[1] && info.bc == BC::Dirichlet) {
            auto coords = fct[fctNo].template get<Coords>();
            for (unsigned q = 0; q < tensor::f::Shape[1]; ++q) {
                auto f = dirichletFun(coords[q]);
                for (unsigned p = 0; p < tensor::f::Shape[0]; ++p) {
                    fview(p, q) = f[p];
                }
            }

            kernel::rhsFacet rhs;
            rhs.c10 = epsilon;
            rhs.c20 = penalty(info);
            rhs.b = b;
            rhs.d_x(0) = d_x;
            rhs.d_xi(0) = d_xi[info.localNo[0]].data();
            rhs.e(0) = e[info.localNo[0]].data();
            rhs.f = f;
            rhs.g(0) = fct[fctNo].template get<JInv>().data()->data();
            rhs.lam_w(0) = userFctPre[fctNo].template get<lam_w_0>().data();
            rhs.mu_w(0) = userFctPre[fctNo].template get<mu_w_0>().data();
            rhs.n = fct[fctNo].template get<Normal>().data()->data();
            rhs.nl = fct[fctNo].template get<NormalLength>().data();
            rhs.w = fctRule.weights().data();
            rhs.execute();

            PetscInt ib = info.g_up[0];
            VecSetValuesBlocked(B, 1, &ib, b, ADD_VALUES);
        }
    }

    CHKERRQ(VecAssemblyBegin(B));
    CHKERRQ(VecAssemblyEnd(B));
    return 0;
}

FiniteElementFunction<DomainDimension> Elasticity::finiteElementFunction(Vec x) const {
    auto numeric = FiniteElementFunction<DomainDimension>(refElement_->clone(), tensor::U::Shape[0],
                                                          tensor::U::Shape[1], numLocalElements());
    PetscScalar const* values;
    VecGetArrayRead(x, &values);
    auto& data = numeric.values();
    auto blockSize = tensor::U::Shape[0] * tensor::U::Shape[1];
    for (std::size_t elNo = 0; elNo < numLocalElements(); ++elNo) {
        auto coords = vol[elNo].template get<Coords>();
        for (std::size_t p = 0; p < tensor::U::Shape[1]; ++p) {
            for (std::size_t dof = 0; dof < tensor::U::Shape[0]; ++dof) {
                data(dof, p, elNo) = values[dof + p * tensor::U::Shape[0] + elNo * blockSize];
            }
        }
    }
    VecRestoreArrayRead(x, &values);
    return numeric;
}

} // namespace tndm
