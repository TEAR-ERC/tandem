#ifndef POISSON_20200627_H
#define POISSON_20200627_H

#include "config.h"
#include "form/DG.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"

#include <petscmat.h>
#include <petscvec.h>

#include <functional>

namespace tndm {

class Poisson : public DG<DomainDimension> {
public:
    using functional_t = std::function<double(std::array<double, DomainDimension> const&)>;

    Poisson(LocalSimplexMesh<DomainDimension> const& mesh, Curvilinear<DomainDimension>& cl,
            std::unique_ptr<RefElement<DomainDimension>> refElement, unsigned minQuadOrder,
            MPI_Comm comm, functional_t kFun);

    Mat assemble();
    Vec rhs(functional_t forceFun, functional_t dirichletFun);

    FiniteElementFunction<DomainDimension> finiteElementFunction(Vec x) const;

    FiniteElementFunction<DomainDimension> discreteK() const {
        return FiniteElementFunction<DomainDimension>(nodalRefElement_.clone(), userVol[0].data(),
                                                      nodalRefElement_.numBasisFunctions(), 1,
                                                      numLocalElements());
    }

private:
    struct K {
        using type = double;
    };

    using user_vol_t = mneme::SingleStorage<K>;
    mneme::StridedView<user_vol_t> userVol;

    Managed<Matrix<double>> Em;
    std::vector<Managed<Matrix<double>>> em;

    NodalRefElement<DomainDimension> nodalRefElement_;

    double penalty(FacetInfo const& info) const {
        auto Kmax = [&](std::size_t elNo) {
            auto nbf = nodalRefElement_.numBasisFunctions();
            auto K = userVol[elNo].data();
            return *std::max_element(K, K + nbf);
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
