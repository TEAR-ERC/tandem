#include "SEAS.h"
#include "common/PetscTimeSolver.h"
#include "config.h"
#include "form/DGOperator.h"
#include "form/DGOperatorTopo.h"
#include "localoperator/Poisson.h"
#include "tandem/Config.h"
#include "tandem/RateAndState.h"
#include "tandem/SeasOperator.h"
#include "tandem/SeasScenario.h"
#include "tandem/SeasWriter.h"

#include "geometry/Curvilinear.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "tensor/Managed.h"

#include <petscsys.h>

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tndm {

void solveSEASProblem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    auto scenario = SeasScenario<tmp::Poisson>(cfg.seas);

    Curvilinear<DomainDimension> cl(mesh, scenario.transform(), PolynomialDegree);

    auto topo = std::make_shared<DGOperatorTopo>(mesh, PETSC_COMM_WORLD);
    auto lop = std::make_unique<tmp::Poisson>(cl, scenario.mu());
    using local_operator_t = typename decltype(lop)::element_type;
    auto quadPoints = lop->facetQuadratureRule().points();
    auto dgop = std::make_unique<DGOperator<local_operator_t>>(topo, std::move(lop));
    using seas_operator_t = SeasOperator<RateAndState, DGOperator<local_operator_t>>;
    auto seasop =
        std::make_shared<seas_operator_t>(topo, std::make_unique<RateAndState>(cl, quadPoints),
                                          std::move(dgop), scenario.ref_normal());
    if (scenario.boundary()) {
        seasop->set_boundary(*scenario.boundary());
    }

    auto ts = PetscTimeSolver(*seasop);

    using seas_writer_t = SeasWriter<DomainDimension, seas_operator_t>;
    std::unique_ptr<seas_writer_t> writer;
    if (cfg.output) {
        writer = std::make_unique<seas_writer_t>(*cfg.output, mesh, cl, seasop, PolynomialDegree);
        ts.set_monitor(*writer);
    }

    ts.solve(cfg.final_time);
}

} // namespace tndm
