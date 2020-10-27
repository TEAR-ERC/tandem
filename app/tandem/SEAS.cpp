#include "SEAS.h"
#include "common/PetscTimeSolver.h"
#include "config.h"
#include "form/DGOperator.h"
#include "form/DGOperatorTopo.h"
#include "localoperator/Poisson.h"
#include "tandem/BP1.h"
#include "tandem/Config.h"
#include "tandem/DieterichRuinaAgeing.h"
#include "tandem/FrictionConfig.h"
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
    using spatial_lop_t = tmp::Poisson;
    using fault_lop_t = DieterichRuinaAgeing;
    using seas_op_t = SeasOperator<RateAndState<fault_lop_t>, DGOperator<spatial_lop_t>>;
    using seas_writer_t = SeasWriter<DomainDimension, seas_op_t>;

    auto scenario = SeasScenario<tmp::Poisson>(cfg.seas);
    auto friction_scenario = DieterichRuinaAgeingScenario(cfg.friction);

    Curvilinear<DomainDimension> cl(mesh, scenario.transform(), PolynomialDegree);

    auto topo = std::make_shared<DGOperatorTopo>(mesh, PETSC_COMM_WORLD);
    auto spatial_lop = std::make_unique<spatial_lop_t>(cl, scenario.mu());
    auto quadPoints = spatial_lop->facetQuadratureRule().points();
    auto dgop = std::make_unique<DGOperator<spatial_lop_t>>(topo, std::move(spatial_lop));
    auto fop = std::make_unique<RateAndState<fault_lop_t>>(cl, quadPoints);
    auto seasop =
        std::make_shared<seas_op_t>(topo, std::move(fop), std::move(dgop), scenario.ref_normal());
    seasop->lop().set_constant_params(friction_scenario.constant_params());
    seasop->lop().set_params(friction_scenario.param_fun());
    if (scenario.boundary()) {
        seasop->set_boundary(*scenario.boundary());
    }

    auto ts = PetscTimeSolver(*seasop);

    std::unique_ptr<seas_writer_t> writer;
    if (cfg.output) {
        writer = std::make_unique<seas_writer_t>(*cfg.output, mesh, cl, seasop, PolynomialDegree);
        ts.set_monitor(*writer);
    }

    ts.solve(cfg.final_time);
}

} // namespace tndm
