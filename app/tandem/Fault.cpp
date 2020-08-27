#include "Fault.h"

#include "BP1.h"
#include "basis/WarpAndBlend.h"
#include "config.h"
#include "kernels/tandem/init.h"
#include "kernels/tandem/kernel.h"
#include "kernels/tandem/tensor.h"
#include "tandem/Elasticity.h"

namespace tndm {

Fault::Fault(LocalSimplexMesh<DomainDimension> const& mesh, Curvilinear<DomainDimension>& cl,
             std::vector<std::array<double, DomainDimension - 1u>> const& quadPoints, MPI_Comm comm)
    : refElement_(PolynomialDegree, WarpAndBlendFactory<DomainDimension - 1u>()), comm_(comm) {

    enodalT = refElement_.evaluateBasisAt(quadPoints, {0, 1});

    auto boundaryData = dynamic_cast<BoundaryData const*>(mesh.facets().data());
    if (!boundaryData) {
        throw std::runtime_error("Boundary conditions not set.");
    }

    std::size_t numFacets = mesh.facets().localSize();
    fctNos_.reserve(numFacets);
    faultNos_.resize(numFacets, std::numeric_limits<std::size_t>::max());
    for (std::size_t fctNo = 0; fctNo < numFacets; ++fctNo) {
        auto bc = boundaryData->getBoundaryConditions()[fctNo];
        if (bc == BC::Fault) {
            faultNos_[fctNo] = fctNos_.size();
            fctNos_.push_back(fctNo);
        }
    }

    auto nbf = refElement_.numBasisFunctions();
    info_.setStorage(std::make_shared<info_t>(fctNos_.size() * nbf), 0u, fctNos_.size(), nbf);

    std::vector<Managed<Matrix<double>>> fctE;
    std::vector<Managed<Tensor<double, 3u>>> fctGradE;
    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto facetParam = cl.facetParam(f, refElement_.refNodes());
        fctE.emplace_back(cl.evaluateBasisAt(facetParam));
        fctGradE.emplace_back(cl.evaluateGradientAt(facetParam));
    }

    sign_.resize(fctNos_.size(), 1.0);
    elNos_.resize(fctNos_.size(), std::numeric_limits<std::size_t>::max());
    localFaceNos_.resize(fctNos_.size(), std::numeric_limits<std::size_t>::max());
#pragma omp parallel
    {

        auto J = Managed(cl.jacobianResultInfo(nbf));
        auto jInv = Managed(cl.jacobianResultInfo(nbf));
        auto normal = Managed(cl.normalResultInfo(nbf));
        auto detJ = Managed(cl.detJResultInfo(nbf));
#pragma omp for
        for (std::size_t faultNo = 0; faultNo < info_.size(); ++faultNo) {
            auto fctNo = fctNos_[faultNo];
            auto elNos = mesh.template upward<DomainDimension - 1u>(fctNo);
            assert(elNos.size() >= 1u && elNos.size() <= 2u);
            auto dws = mesh.template downward<DomainDimension - 1u, DomainDimension>(elNos[0]);
            auto localFaceNo = std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fctNo));
            assert(localFaceNo < DomainDimension + 1u);

            elNos_[faultNo] = elNos[0];
            localFaceNos_[faultNo] = localFaceNo;

            auto coords =
                Tensor(info_[faultNo].template get<Coords>().data()->data(), cl.mapResultInfo(nbf));
            cl.map(elNos[0], fctE[localFaceNo], coords);

            auto R = Tensor<double, 3u>(info_[faultNo].template get<Rnodal>().data()->data(),
                                        DomainDimension, DomainDimension, nbf);
            cl.jacobian(elNos[0], fctGradE[localFaceNo], J);
            cl.detJ(elNos[0], J, detJ);
            cl.jacobianInv(J, jInv);
            cl.normal(localFaceNo, detJ, jInv, normal);
            if (normal(0, 0) < 0.0) {
                sign_[faultNo] = -1.0;
            }
            cl.facetBasis(localFaceNo, J, normal, R);
        }
    }
}

PetscErrorCode Fault::createState(Vec* state) const {
    PetscInt blockSize_ = 2 * refElement_.numBasisFunctions();
    PetscInt localRows = info_.size() * blockSize_;
    CHKERRQ(VecCreate(comm_, state));
    CHKERRQ(VecSetSizes(*state, localRows, PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(*state));
    return 0;
}

PetscErrorCode Fault::initial(Vec state) const {
    BP1 bp1;

    PetscScalar* Xraw;
    VecGetArray(state, &Xraw);
    auto X = tensor(Xraw);
    std::size_t nbf = refElement_.numBasisFunctions();
    for (std::size_t faultNo = 0; faultNo < info_.size(); ++faultNo) {
        auto coords = info_[faultNo].get<Coords>();
        for (std::size_t node = 0; node < nbf; ++node) {
            bp1.setX(coords[node]);
            X(node, 0, faultNo) = bp1.psi0();
            X(node, 1, faultNo) = 0.0;
        }
    }
    VecRestoreArray(state, &Xraw);
    CHKERRQ(VecAssemblyBegin(state));
    CHKERRQ(VecAssemblyEnd(state));
    return 0;
}

void Fault::rhs(Elasticity const& elasticity, Vec u, Vec x, Vec f) const {
    PetscScalar const* Xraw;
    PetscScalar const* U;
    PetscScalar* Fraw;
    VecGetArrayRead(x, &Xraw);
    VecGetArrayRead(u, &U);
    VecGetArray(f, &Fraw);

    auto X = tensor(Xraw);
    auto F = tensor(Fraw);

    std::size_t nbf = refElement_.numBasisFunctions();
#pragma omp parallel
    {
        BP1 bp1;
        double tractionBuf[tensor::traction_avg_proj::size()];
        auto traction = Matrix<double>(tractionBuf, tensor::traction_avg_proj::Shape[0],
                                       tensor::traction_avg_proj::Shape[1]);
#pragma omp for
        for (std::size_t faultNo = 0; faultNo < info_.size(); ++faultNo) {
            auto coords = info_[faultNo].get<Coords>();
            double const* R = info_[faultNo].get<Rnodal>().data()->data();
            elasticity.traction(fctNos_[faultNo], R, U, traction);
            for (std::size_t node = 0; node < nbf; ++node) {
                bp1.setX(coords[node]);
                PetscInt row = node + 2 * faultNo * nbf;
                if (coords[node][1] <= -40000.0) {
                    F(node, 0, faultNo) = 0.0;
                    F(node, 1, faultNo) = 1e-9;
                } else {
                    double tau = std::copysign(traction(1, node), sign_[faultNo]);
                    double psi = X(node, 0, faultNo);
                    double V = bp1.computeSlipRate(tau, psi);
                    F(node, 0, faultNo) = bp1.G(tau, V, psi);
                    F(node, 1, faultNo) = V;
                }
            }
        }
    }

    VecRestoreArray(f, &Fraw);
    VecRestoreArrayRead(u, &U);
    VecRestoreArrayRead(x, &Xraw);
    VecAssemblyBegin(f);
    VecAssemblyEnd(f);
}

auto Fault::slip(Vec x) const -> facet_functional_t {
    std::size_t nbf = refElement_.numBasisFunctions();
    return [this, nbf, x](std::size_t fctNo, double* f) {
        double const* Xraw;
        VecGetArrayRead(x, &Xraw);
        auto X = tensor(Xraw);
        auto faultNo = this->faultNos_[fctNo];
        double g[tensor::slip_proj::size()];
        auto gview = init::slip_proj::view::create(g);
        for (std::size_t i = 0; i < nbf; ++i) {
            gview(0, i) = 0.0;
            gview(1, i) = std::copysign(X(i, 1, faultNo), this->sign_[faultNo]);
        }
        VecRestoreArrayRead(x, &Xraw);

        double const* R = info_[faultNo].get<Rnodal>().data()->data();
        kernel::evaluate_slip krnl;
        krnl.slip_proj = g;
        krnl.enodalT = this->enodalT.data();
        krnl.f = f;
        krnl.Rnodal = R;
        krnl.execute();
    };
}

} // namespace tndm
