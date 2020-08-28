#ifndef FAULT_20200825_H
#define FAULT_20200825_H

#include "config.h"

#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "mesh/LocalSimplexMesh.h"
#include "tandem/Elasticity.h"

#include <mneme/storage.hpp>

#include <mpi.h>
#include <petscsystypes.h>
#include <petscts.h>
#include <petscvec.h>

namespace tndm {

class Fault {
public:
    using facet_functional_t = Elasticity::facet_functional_t;

    Fault(LocalSimplexMesh<DomainDimension> const& mesh, Curvilinear<DomainDimension>& cl,
          std::vector<std::array<double, DomainDimension - 1u>> const& quadPoints, MPI_Comm comm);

    PetscErrorCode createState(Vec* state) const;
    PetscErrorCode initial(Vec state) const;

    void rhs(Elasticity const& elasticity, Vec u, Vec x, Vec f) const;

    auto slip(Vec x) const -> facet_functional_t;

    std::vector<std::size_t> const& elNos() const { return elNos_; }
    std::vector<std::size_t> const& localFaceNos() const { return localFaceNos_; }

    auto tensor(double* state) const {
        return Tensor<double, 3u>(state, refElement_.numBasisFunctions(), 2, fctNos_.size());
    }
    auto tensor(double const* state) const {
        return Tensor<double const, 3u>(state, refElement_.numBasisFunctions(), 2, fctNos_.size());
    }

    auto finiteElementFunction(Vec x) const {
        double const* Xraw;
        VecGetArrayRead(x, &Xraw);
        // auto fun = FiniteElementFunction<DomainDimension - 1u>(
        // refElement_.clone(), Xraw, refElement_.numBasisFunctions(), 2, fctNos_.size());
        auto X = tensor(Xraw);
        auto fun = FiniteElementFunction<DomainDimension - 1u>(
            refElement_.clone(), refElement_.numBasisFunctions(), 4, fctNos_.size());
        for (std::size_t faultNo = 0; faultNo < info_.size(); ++faultNo) {
            auto tau = info_[faultNo].get<Tau>();
            auto V = info_[faultNo].get<SlipRate>();
            for (std::size_t node = 0; node < refElement_.numBasisFunctions(); ++node) {
                fun.values()(node, 0, faultNo) = X(node, 0, faultNo);
                fun.values()(node, 1, faultNo) = X(node, 1, faultNo);
                fun.values()(node, 2, faultNo) = tau[node];
                fun.values()(node, 3, faultNo) = V[node];
            }
        }
        VecRestoreArrayRead(x, &Xraw);
        return fun;
    }

private:
    NodalRefElement<DomainDimension - 1u> refElement_;
    MPI_Comm comm_;
    std::vector<std::size_t> fctNos_;
    std::vector<std::size_t> faultNos_;
    std::vector<std::size_t> elNos_;
    std::vector<std::size_t> localFaceNos_;
    std::vector<double> sign_;

    Managed<Matrix<double>> enodalT;

    struct Coords {
        using type = std::array<double, DomainDimension>;
    };
    struct Rnodal {
        using type = std::array<double, DomainDimension * DomainDimension>;
    };
    struct Tau {
        using type = double;
    };
    struct SlipRate {
        using type = double;
    };

    using info_t = mneme::MultiStorage<mneme::DataLayout::SoA, Coords, Rnodal, Tau, SlipRate>;
    mneme::StridedView<info_t> info_;
};

} // namespace tndm

#endif // FAULT_20200825_H
