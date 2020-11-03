#include "SEAS.h"
#include "common/PetscTimeSolver.h"
#include "config.h"
#include "localoperator/Poisson.h"
#include "tandem/Config.h"
#include "tandem/DieterichRuinaAgeing.h"
#include "tandem/FrictionConfig.h"
#include "tandem/RateAndState.h"
#include "tandem/SeasOperator.h"
#include "tandem/SeasPoissonAdapter.h"
#include "tandem/SeasScenario.h"
#include "tandem/SeasWriter.h"

#include "form/DGOperatorTopo.h"
#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "tensor/Managed.h"

#include <mpi.h>
#include <petscsys.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tndm {

void solveSEASProblem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    using fault_lop_t = DieterichRuinaAgeing;
    using seas_op_t = SeasOperator<RateAndState<fault_lop_t>, SeasPoissonAdapter>;
    using seas_writer_t = SeasWriter<DomainDimension, seas_op_t>;

    auto scenario = SeasScenario<Poisson>(cfg.seas);
    auto friction_scenario = DieterichRuinaAgeingScenario(cfg.friction);

    auto cl = std::make_shared<Curvilinear<DomainDimension>>(mesh, scenario.transform(),
                                                             PolynomialDegree);

    auto fop = std::make_unique<RateAndState<fault_lop_t>>(cl);
    auto topo = std::make_shared<DGOperatorTopo>(mesh, PETSC_COMM_WORLD);
    auto spatial_lop = std::make_unique<Poisson>(cl, scenario.mu());
    auto adapter =
        std::make_unique<SeasPoissonAdapter>(cl, topo, fop->space().clone(), std::move(spatial_lop),
                                             scenario.ref_normal(), cfg.seas.normal_stress);

    auto seasop = std::make_shared<seas_op_t>(std::move(fop), std::move(adapter));
    seasop->lop().set_constant_params(friction_scenario.constant_params());
    seasop->lop().set_params(friction_scenario.param_fun());
    if (friction_scenario.source_fun()) {
        seasop->lop().set_source_fun(*friction_scenario.source_fun());
    }
    if (scenario.boundary()) {
        seasop->set_boundary(*scenario.boundary());
    }

    auto ts = PetscTimeSolver(*seasop);

    std::unique_ptr<seas_writer_t> writer;
    if (cfg.output) {
        writer = std::make_unique<seas_writer_t>(cfg.output->prefix, mesh, cl, seasop,
                                                 PolynomialDegree, cfg.output->V_ref,
                                                 cfg.output->t_min, cfg.output->t_max);
        ts.set_monitor(*writer);
    }

    ts.solve(cfg.final_time);

    auto solution = scenario.solution(cfg.final_time);
    if (solution) {
        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        auto numeric = seasop->adapter().displacement();
        double error =
            tndm::Error<DomainDimension>::L2(*cl, numeric, *solution, 0, PETSC_COMM_WORLD);
        if (rank == 0) {
            std::cout << "L2 error: " << error << std::endl;
        }
    }
}

} // namespace tndm
