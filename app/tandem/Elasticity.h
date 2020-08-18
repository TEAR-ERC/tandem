#ifndef ELASTICITY_20200723_H
#define ELASTICITY_20200723_H

#include "common/InterfacePetsc.h"
#include "config.h"
#include "form/DG.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "mneme/storage.hpp"

#include <petscmat.h>
#include <petscvec.h>

#include <functional>

namespace tndm {

class Elasticity : public DG<DomainDimension> {
public:
    using functional_t = std::function<double(std::array<double, DomainDimension> const&)>;
    using vector_functional_t = std::function<std::array<double, DomainDimension>(
        std::array<double, DomainDimension> const&)>;

    Elasticity(LocalSimplexMesh<DomainDimension> const& mesh, Curvilinear<DomainDimension>& cl,
               std::unique_ptr<RefElement<DomainDimension>> refElement, unsigned minQuadOrder,
               MPI_Comm comm, functional_t const& lambdaFun, functional_t const& muFun);

    InterfacePetsc interfacePetsc() {
        auto blockSize = refElement_->numBasisFunctions() * DomainDimension;
        return InterfacePetsc(blockSize, numLocalElements(), &volInfo[0].get<NumLocalNeighbours>(),
                              &volInfo[0].get<NumGhostNeighbours>(), comm());
    }

    PetscErrorCode createAShell(Mat* A);

    PetscErrorCode assemble(Mat mat);
    PetscErrorCode rhs(Vec B, vector_functional_t forceFun, vector_functional_t dirichletFun,
                       vector_functional_t slipFun);

    FiniteElementFunction<DomainDimension> finiteElementFunction(Vec x) const;

    static PetscErrorCode ApplyDGOperator(Mat mat, Vec x, Vec y) {
        void* ptr;
        MatShellGetContext(mat, &ptr);
        Elasticity* user = reinterpret_cast<Elasticity*>(ptr);
        PetscScalar const* xraw;
        PetscScalar* yraw;
        VecGetArrayRead(x, &xraw);
        VecGetArray(y, &yraw);
        user->apply(xraw, yraw);
        VecRestoreArray(y, &yraw);
        VecRestoreArrayRead(x, &xraw);
        VecAssemblyBegin(y);
        VecAssemblyEnd(y);
        return 0;
    }

    void apply(double const* U, double* Unew);

    FiniteElementFunction<DomainDimension> discreteLambda() const { return discreteField<lam>(); }
    FiniteElementFunction<DomainDimension> discreteMu() const { return discreteField<mu>(); }

private:
    template <typename T> FiniteElementFunction<DomainDimension> discreteField() const {
        return FiniteElementFunction<DomainDimension>(
            nodalRefElement_.clone(), userVol[0].get<T>().data(),
            nodalRefElement_.numBasisFunctions(), 1, numLocalElements());
    }

    struct lam {
        using type = double;
    };
    struct mu {
        using type = double;
    };
    struct lam_W_J {
        using type = double;
    };
    struct mu_W_J {
        using type = double;
    };
    struct lam_w_0 {
        using type = double;
    };
    struct mu_w_0 {
        using type = double;
    };
    struct lam_w_1 {
        using type = double;
    };
    struct mu_w_1 {
        using type = double;
    };

    using user_vol_t = mneme::MultiStorage<mneme::DataLayout::SoA, lam, mu>;
    mneme::StridedView<user_vol_t> userVol;

    using user_vol_pre_t = mneme::MultiStorage<mneme::DataLayout::SoA, lam_W_J, mu_W_J>;
    mneme::StridedView<user_vol_pre_t> userVolPre;

    using user_fct_pre_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, lam_w_0, mu_w_0, lam_w_1, mu_w_1>;
    mneme::StridedView<user_fct_pre_t> userFctPre;

    NodalRefElement<DomainDimension> nodalRefElement_;

    double penalty(FacetInfo const& info) const {
        auto lambdaMax = [&](std::size_t elNo) {
            auto nbf = nodalRefElement_.numBasisFunctions();
            auto field = userVol[elNo].get<lam>().data();
            return *std::max_element(field, field + nbf);
        };
        auto muMax = [&](std::size_t elNo) {
            auto nbf = nodalRefElement_.numBasisFunctions();
            auto field = userVol[elNo].get<mu>().data();
            return *std::max_element(field, field + nbf);
        };
        double maxMat = std::max(std::max(muMax(info.up[0]), muMax(info.up[1])),
                                 std::max(lambdaMax(info.up[0]), lambdaMax(info.up[1])));
        double penaltyScale =
            (PolynomialDegree + 1) * (PolynomialDegree + DomainDimension) / DomainDimension;
        return penaltyScale * maxMat *
               std::max(volInfo[info.up[0]].get<Penalty>(), volInfo[info.up[1]].get<Penalty>());
    }

    double epsilon = -1.0;
};

} // namespace tndm

#endif // ELASTICITY_20200723_H
