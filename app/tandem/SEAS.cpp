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
#include "io/VTUAdapter.h"
#include "io/VTUWriter.h"
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
    auto seasop = SeasOperator(topo, std::make_unique<RateAndState>(cl));

    // PetscLinearSolver ls(seasop);
    auto ts = PetscTimeSolver(seasop);

    if (cfg.output) {
        auto fault_writer =
            VTUWriter<DomainDimension - 1u>(PolynomialDegree, true, PETSC_COMM_WORLD);
        auto fault_adapter = CurvilinearBoundaryVTUAdapter(mesh, cl, seasop.faultMap().fctNos());
        auto fault_piece = fault_writer.addPiece(fault_adapter);
        fault_piece.addPointData("state", seasop.state(ts.state()));
        fault_writer.write(*cfg.output);
    }

    // seasop.setup_quasi_dynamic(ls, ts);
}

} // namespace tndm
