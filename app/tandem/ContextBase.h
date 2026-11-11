#ifndef CONTEXTBASE_20210910_H
#define CONTEXTBASE_20210910_H

#include "config.h"
#include "form/AbstractAdapterOperator.h"
#include "form/AbstractFrictionOperator.h"
#include "form/SeasFDOperator.h"
#include "form/SeasQDOperator.h"

#include "form/AbstractDGOperator.h"
#include "form/BoundaryMap.h"
#include "form/DGOperatorTopo.h"
#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "mesh/LocalSimplexMesh.h"

#include <petscsys.h>

namespace tndm::seas {

class ContextBase {
public:
    using transform_t = Curvilinear<DomainDimension>::transform_t;

    virtual ~ContextBase() {}

    /**
     * @param comm Communicator the mesh is distributed over. This is PETSC_COMM_WORLD, except for
     *             the workers that compute Green's functions on a sub-communicator.
     */
    ContextBase(LocalSimplexMesh<DomainDimension> const& mesh, transform_t transform,
                MPI_Comm comm = PETSC_COMM_WORLD)
        : cl(std::make_shared<Curvilinear<DomainDimension>>(mesh, std::move(transform),
                                                            PolynomialDegree)),
          fault_map(std::make_shared<BoundaryMap>(mesh, BC::Fault, comm)),
          topo(std::make_shared<DGOperatorTopo>(mesh, comm)) {}

    virtual auto dg() -> std::unique_ptr<AbstractDGOperator<DomainDimension>> = 0;
    virtual auto friction() -> std::unique_ptr<AbstractFrictionOperator> = 0;
    virtual auto adapter() -> std::unique_ptr<AbstractAdapterOperator> = 0;
    virtual void setup_seasop(SeasQDOperator& seasop) = 0;
    virtual void setup_seasop(SeasFDOperator& seasop) = 0;
    virtual auto domain_solution(double time) -> std::unique_ptr<SolutionInterface> = 0;
    virtual auto fault_solution(double time) -> std::unique_ptr<SolutionInterface> = 0;

    std::shared_ptr<Curvilinear<DomainDimension>> cl;
    std::shared_ptr<BoundaryMap> fault_map;
    std::shared_ptr<DGOperatorTopo> topo;
};

} // namespace tndm::seas

#endif // CONTEXTBASE_20210910_H
