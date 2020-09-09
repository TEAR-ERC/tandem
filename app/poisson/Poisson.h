#ifndef POISSON_20200627_H
#define POISSON_20200627_H

#include "common/InterfacePetsc.h"
#include "config.h"
#include "form/DG.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"

#include "mneme/span.hpp"
#include "mneme/storage.hpp"
#include "mneme/view.hpp"
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace tndm {

class Poisson : public DG<DomainDimension> {
public:
    using functional_t = std::function<double(std::array<double, DomainDimension> const&)>;
    using volume_functional_t = std::function<void(std::size_t elNo, double* F)>;
    using facet_functional_t = std::function<void(std::size_t fctNo, double* f)>;

    Poisson(LocalSimplexMesh<DomainDimension> const& mesh, Curvilinear<DomainDimension>& cl,
            std::unique_ptr<RefElement<DomainDimension>> refElement, unsigned minQuadOrder,
            MPI_Comm comm, functional_t kFun);

    InterfacePetsc interfacePetsc() {
        return InterfacePetsc(refElement_->numBasisFunctions(), numLocalElements(),
                              &volInfo[0].get<NumLocalNeighbours>(),
                              &volInfo[0].get<NumGhostNeighbours>(), comm());
    }

    PetscErrorCode assemble(Mat mat) const;
    PetscErrorCode rhs(Vec B, volume_functional_t forceFun, facet_functional_t dirichletFun,
                       facet_functional_t slipFun) const;
    PetscErrorCode rhs(Vec B, functional_t forceFun, functional_t dirichletFun,
                       functional_t slipFun) const {
        return rhs(B, makeVolumeFunctional(forceFun), makeFacetFunctional(dirichletFun),
                   makeFacetFunctional(slipFun));
    }

    volume_functional_t makeVolumeFunctional(functional_t fun) const;
    facet_functional_t makeFacetFunctional(functional_t fun) const;

    void grad_u(std::size_t fctNo, double const* U, Matrix<double>& result) const;

    FiniteElementFunction<DomainDimension> finiteElementFunction(Vec x) const;

    FiniteElementFunction<DomainDimension> discreteK() const {
        return FiniteElementFunction<DomainDimension>(
            nodalRefElement_.clone(), userVol[0].get<K>().data(),
            nodalRefElement_.numBasisFunctions(), 1, numLocalElements());
    }

private:
    struct K {
        using type = double;
    };

    using user_vol_t = mneme::MultiStorage<mneme::DataLayout::SoA, K>;
    mneme::StridedView<user_vol_t> userVol;

    Managed<Matrix<double>> Em;
    std::vector<Managed<Matrix<double>>> em;

    NodalRefElement<DomainDimension> nodalRefElement_;
    NodalRefElement<DomainDimension - 1u> facetRefElement_;
    Managed<Matrix<double>> minv;
    Managed<Matrix<double>> enodal;

    double penalty(FacetInfo const& info) const {
        auto Kmax = [&](std::size_t elNo) {
            auto nbf = nodalRefElement_.numBasisFunctions();
            auto Kfield = userVol[elNo].get<K>().data();
            return *std::max_element(Kfield, Kfield + nbf);
        };
        double penaltyScale =
            (PolynomialDegree + 1) * (PolynomialDegree + DomainDimension) / DomainDimension;
        return penaltyScale *
               std::max(volInfo[info.up[0]].get<Penalty>(), volInfo[info.up[1]].get<Penalty>()) *
               std::max(Kmax(info.up[0]), Kmax(info.up[1]));
    }

    double epsilon = -1.0;
};

} // namespace tndm

#endif // POISSON_20200627_H
