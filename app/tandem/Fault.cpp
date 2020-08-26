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
    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        fctE.emplace_back(cl.evaluateBasisAt(cl.facetParam(f, refElement_.refNodes())));
    }

#pragma omp parallel for
    for (std::size_t faultNo = 0; faultNo < info_.size(); ++faultNo) {
        auto fctNo = fctNos_[faultNo];
        auto elNos = mesh.template upward<DomainDimension - 1u>(fctNo);
        assert(elNos.size() >= 1u && elNos.size() <= 2u);
        auto dws = mesh.template downward<DomainDimension - 1u, DomainDimension>(elNos[0]);
        auto localFctNo = std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fctNo));
        assert(localFctNo < DomainDimension + 1u);

        auto coords =
            Tensor(info_[faultNo].template get<Coords>().data()->data(), cl.mapResultInfo(nbf));
        cl.map(elNos[0], fctE[localFctNo], coords);
    }
}

PetscErrorCode Fault::createState(Vec* state) const {
    PetscInt blockSize_ = 2 * refElement_.numBasisFunctions();
    PetscInt localRows = info_.size() * blockSize_;
    CHKERRQ(VecCreate(comm_, state));
    CHKERRQ(VecSetSizes(*state, localRows, PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(*state));
    CHKERRQ(VecSetBlockSize(*state, blockSize_));
    return 0;
}

PetscErrorCode Fault::initial(Vec psi) const {
    BP1 bp1;

    std::size_t nbf = refElement_.numBasisFunctions();
    for (std::size_t faultNo = 0; faultNo < info_.size(); ++faultNo) {
        auto coords = info_[faultNo].get<Coords>();
        for (std::size_t node = 0; node < nbf; ++node) {
            bp1.setX(coords[node]);
            PetscInt row = node + 2 * faultNo * nbf;
            VecSetValue(psi, row, bp1.psi0(), INSERT_VALUES);
            VecSetValue(psi, row + nbf, 0.0, INSERT_VALUES);
        }
    }
    CHKERRQ(VecAssemblyBegin(psi));
    CHKERRQ(VecAssemblyEnd(psi));
    return 0;
}

void Fault::rhs(Elasticity const& elasticity, Vec u, Vec x, Vec f) const {
    PetscScalar const* X;
    PetscScalar const* U;
    PetscScalar* F;
    VecGetArrayRead(x, &X);
    VecGetArrayRead(u, &U);
    VecGetArray(f, &F);

    std::size_t nbf = refElement_.numBasisFunctions();
#pragma omp parallel
    {
        BP1 bp1;
        double traction[tensor::traction_avg_proj::size()];
        auto tau = Matrix<double>(traction, tensor::traction_avg_proj::Shape[0],
                                  tensor::traction_avg_proj::Shape[1]);
#pragma omp for
        for (std::size_t faultNo = 0; faultNo < info_.size(); ++faultNo) {
            auto coords = info_[faultNo].get<Coords>();
            elasticity.traction(fctNos_[faultNo], U, tau);
            for (std::size_t node = 0; node < nbf; ++node) {
                bp1.setX(coords[node]);
                PetscInt row = node + 2 * faultNo * nbf;
                if (coords[node][1] <= -40000.0) {
                    F[row] = 0.0;
                    F[row + nbf] = 1e-9;
                } else {
                    double V = bp1.computeSlipRate(tau(1, node), X[row]);
                    F[row] = bp1.G(tau(1, node), V, X[row]);
                    F[row + nbf] = V;
                }
            }
        }
    }

    VecRestoreArray(f, &F);
    VecRestoreArrayRead(u, &U);
    VecRestoreArrayRead(x, &X);
    VecAssemblyBegin(f);
    VecAssemblyEnd(f);
}

auto Fault::slip(Vec x) const -> facet_functional_t {
    std::size_t nbf = refElement_.numBasisFunctions();
    return [this, nbf, x](std::size_t fctNo, double* f) {
        double const* X;
        VecGetArrayRead(x, &X);
        auto faultNo = this->faultNos_[fctNo];
        double g[tensor::slip_proj::size()];
        auto gview = init::slip_proj::view::create(g);
        for (std::size_t i = 0; i < nbf; ++i) {
            gview(0, i) = 0.0;
            gview(1, i) = X[i + nbf + 2 * faultNo * nbf];
        }
        VecRestoreArrayRead(x, &X);

        kernel::evaluate_slip krnl;
        krnl.slip_proj = g;
        krnl.enodalT = this->enodalT.data();
        krnl.f = f;
        krnl.execute();
    };
}

} // namespace tndm
