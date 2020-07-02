#ifndef POISSON_20200627_H
#define POISSON_20200627_H

#include "config.h"
#include "form/DG.h"
#include "form/FiniteElementFunction.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <functional>

namespace tndm {

class Poisson : public DG<DomainDimension> {
public:
    using functional_t = std::function<double(std::array<double, DomainDimension> const&)>;

    using DG<DomainDimension>::DG;

    Eigen::SparseMatrix<double> assemble();

    Eigen::VectorXd rhs(functional_t forceFun, functional_t dirichletFun);

    FiniteElementFunction<DomainDimension>
    finiteElementFunction(Eigen::VectorXd const& numeric) const;

private:
    double penalty(FacetInfo const& info) const {
        double penaltyScale =
            (PolynomialDegree + 1) * (PolynomialDegree + DomainDimension) / DomainDimension;
        return penaltyScale * std::max(penalty_[info.up[0]], penalty_[info.up[1]]);
    }

    double epsilon = -1.0;
};

} // namespace tndm

#endif // POISSON_20200627_H
