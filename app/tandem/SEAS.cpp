#include "SEAS.h"
#include "common/PetscSolver.h"
#include "common/PetscTS.h"
#include "common/PoissonScenario.h"
#include "config.h"
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

    auto lop = scenario.make_local_operator(cl);
    auto seasop = SeasOperator<DomainDimension, tmp::Poisson, RateAndState>(
        mesh, std::move(lop), std::make_unique<RateAndState>(cl), PETSC_COMM_WORLD);

    PetscSolver ls;
    PetscTS ts;

    seasop.setup_quasi_dynamic(ls, ts);
}

} // namespace tndm
