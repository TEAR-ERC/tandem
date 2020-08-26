#ifndef FAULT_20200825_H
#define FAULT_20200825_H

#include "config.h"

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

private:
    NodalRefElement<DomainDimension - 1u> refElement_;
    MPI_Comm comm_;
    std::vector<std::size_t> fctNos_;
    std::vector<std::size_t> faultNos_;

    Managed<Matrix<double>> enodalT;

    struct Coords {
        using type = std::array<double, DomainDimension>;
    };

    using info_t = mneme::MultiStorage<mneme::DataLayout::SoA, Coords>;
    mneme::StridedView<info_t> info_;
};

} // namespace tndm

#endif // FAULT_20200825_H
