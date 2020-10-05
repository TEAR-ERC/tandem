#include "SEAS.h"
#include "common/PetscLinearSolver.h"
#include "common/PetscTimeSolver.h"
#include "common/PoissonScenario.h"
#include "config.h"
#include "form/DGOperatorTopo.h"
#include "localoperator/Poisson.h"
#include "tandem/Config.h"
#include "tandem/RateAndState.h"
#include "tandem/SeasOperator.h"

#include "geometry/Curvilinear.h"
#include "tensor/Managed.h"

#include <petscsys.h>

#include <memory>
#include <utility>

namespace tndm {

void solveSEASProblem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    auto scenario = PoissonScenario(cfg.poisson);

    Curvilinear<DomainDimension> cl(mesh, scenario.transform(), PolynomialDegree);

    auto topo = std::make_shared<DGOperatorTopo>(mesh, PETSC_COMM_WORLD);
    auto lop = scenario.make_local_operator(cl);
    auto seasop = SeasOperator<tmp::Poisson, RateAndState>(topo, std::move(lop),
                                                           std::make_unique<RateAndState>(cl));

    // PetscLinearSolver ls(seasop);
    // PetscTimeSolver ts;

    // seasop.setup_quasi_dynamic(ls, ts);
}

} // namespace tndm
