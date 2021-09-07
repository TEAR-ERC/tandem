#include "SEAS.h"
#include "common/PetscTimeSolver.h"
#include "config.h"
#include "form/AdapterOperator.h"
#include "form/BoundaryMap.h"
#include "form/FrictionOperator.h"
#include "form/SeasQDOperator.h"
#include "localoperator/DieterichRuinaAgeing.h"
#include "localoperator/Elasticity.h"
#include "localoperator/ElasticityAdapter.h"
#include "localoperator/Poisson.h"
#include "localoperator/PoissonAdapter.h"
#include "localoperator/RateAndState.h"
#include "tandem/FrictionConfig.h"
#include "tandem/SeasScenario.h"
#include "tandem/SeasWriter.h"

#include "form/DGOperator.h"
#include "form/DGOperatorTopo.h"
#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "parallel/MPITraits.h"
#include "tensor/Managed.h"
#include "util/Stopwatch.h"

#include <mpi.h>
#include <petscsys.h>

#include <algorithm>
#include <array>
#include <ctime>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tndm::detail {

template <SeasType Type> struct make_lop;
template <> struct make_lop<SeasType::Poisson> {
    using type = Poisson;
    using adapter_type = PoissonAdapter;

    static auto dg(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                   SeasScenario<type> const& scenario) {
        return std::make_unique<type>(std::move(cl), scenario.mu(), DGMethod::IP);
    }

    static auto adapter(Config const& cfg, std::shared_ptr<Curvilinear<DomainDimension>> cl,
                        std::unique_ptr<RefElement<DomainDimension - 1u>> space,
                        SimplexQuadratureRule<DomainDimension - 1u> const& quad_rule) {
        return std::make_unique<PoissonAdapter>(std::move(cl), std::move(space), quad_rule, cfg.up,
                                                cfg.ref_normal);
    }
};
template <> struct make_lop<SeasType::Elasticity> {
    using type = Elasticity;
    using adapter_type = ElasticityAdapter;

    static auto dg(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                   SeasScenario<type> const& scenario) {
        return std::make_unique<type>(std::move(cl), scenario.lam(), scenario.mu(), DGMethod::IP);
    }

    static auto adapter(Config const& cfg, std::shared_ptr<Curvilinear<DomainDimension>> cl,
                        std::unique_ptr<RefElement<DomainDimension - 1u>> space,
                        SimplexQuadratureRule<DomainDimension - 1u> const& quad_rule) {
        return std::make_unique<ElasticityAdapter>(std::move(cl), std::move(space), quad_rule,
                                                   cfg.up, cfg.ref_normal);
    }
};

// template <class Adapter, bool MakeGreen> struct discrete_green;
// template <class Adapter> struct discrete_green<Adapter, false> {
// using type = Adapter;
// static auto wrap(std::unique_ptr<Adapter> adapter, std::size_t) { return std::move(adapter); }
//};
// template <class Adapter> struct discrete_green<Adapter, true> {
// using type = DiscreteGreenAdapter<Adapter>;
// static auto wrap(std::unique_ptr<Adapter> adapter, std::size_t slip_block_size) {
// return std::make_unique<type>(std::move(adapter), slip_block_size);
//}
//};

template <SeasType Type, bool MakeGreen>
void solve_seas_problem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    using adapter_lop_t = typename make_lop<Type>::adapter_type;
    using adapter_t = AdapterOperator<adapter_lop_t>;
    using dg_lop_t = typename make_lop<Type>::type;
    using dg_t = DGOperator<dg_lop_t>;
    using friction_lop_t = RateAndState<DieterichRuinaAgeing>;
    using friction_t = FrictionOperator<friction_lop_t>;
    using seas_t = SeasQDOperator<adapter_t, dg_t, friction_t>;
    using seas_fault_probe_writer_t = SeasFaultProbeWriter<DomainDimension, seas_t>;
    using seas_domain_probe_writer_t = SeasDomainProbeWriter<DomainDimension, seas_t>;
    using seas_fault_writer_t = SeasFaultWriter<DomainDimension, seas_t>;
    using seas_fault_scalar_writer_t = SeasFaultScalarWriter<seas_t>;
    using seas_domain_writer_t = SeasDomainWriter<DomainDimension, seas_t>;
    using seas_monitor_t = SeasMonitor<seas_t>;

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    auto scenario = SeasScenario<dg_lop_t>(cfg.lib, cfg.scenario);
    auto friction_scenario = DieterichRuinaAgeingScenario(cfg.lib, cfg.scenario);

    auto cl = std::make_shared<Curvilinear<DomainDimension>>(mesh, scenario.transform(),
                                                             PolynomialDegree);
    auto fault_map = std::make_shared<BoundaryMap>(mesh, BC::Fault, PETSC_COMM_WORLD);
    auto topo = std::make_shared<DGOperatorTopo>(mesh, PETSC_COMM_WORLD);

    auto dgop = std::make_unique<dg_t>(topo, make_lop<Type>::dg(cl, scenario));

    auto friction =
        std::make_unique<friction_t>(std::make_unique<friction_lop_t>(cl), topo, fault_map);
    friction->lop().set_constant_params(friction_scenario.constant_params());
    friction->lop().set_params(friction_scenario.param_fun());
    if (friction_scenario.source_fun()) {
        friction->lop().set_source_fun(*friction_scenario.source_fun());
    }

    auto adapter = std::make_unique<adapter_t>(
        make_lop<Type>::adapter(cfg, cl, friction->lop().space().clone(),
                                dgop->lop().facetQuadratureRule()),
        topo, fault_map);
    auto seasop =
        std::make_shared<seas_t>(std::move(dgop), std::move(adapter), std::move(friction),
                                 cfg.matrix_free, MGConfig(cfg.mg_coarse_level, cfg.mg_strategy));
    if (scenario.boundary()) {
        seasop->set_boundary(*scenario.boundary());
    }
    seasop->warmup();

    auto ts = PetscTimeSolver(*seasop);

    std::vector<std::unique_ptr<SeasWriter>> writers;
    {
        if (cfg.fault_output && cfg.domain_output) {
            if (cfg.fault_output->prefix == cfg.domain_output->prefix) {
                throw std::runtime_error(
                    "Fault output prefix and domain output prefix must not be identical");
            }
        }
        if (cfg.fault_probe_output) {
            writers.emplace_back(std::make_unique<seas_fault_probe_writer_t>(
                cfg.fault_probe_output->prefix, cfg.fault_probe_output->probes,
                cfg.fault_probe_output->make_adaptive_output_interval(), mesh, cl, seasop));
        }
        if (cfg.domain_probe_output) {
            writers.emplace_back(std::make_unique<seas_domain_probe_writer_t>(
                cfg.domain_probe_output->prefix, cfg.domain_probe_output->probes,
                cfg.domain_probe_output->make_adaptive_output_interval(), mesh, cl, seasop));
        }
        if (cfg.fault_output) {
            writers.emplace_back(std::make_unique<seas_fault_writer_t>(
                cfg.fault_output->prefix, cfg.fault_output->make_adaptive_output_interval(), mesh,
                cl, seasop, PolynomialDegree));
        }
        if (cfg.fault_scalar_output) {
            writers.emplace_back(std::make_unique<seas_fault_scalar_writer_t>(
                cfg.fault_scalar_output->prefix,
                cfg.fault_scalar_output->make_adaptive_output_interval(), seasop));
        }
        if (cfg.domain_output) {
            writers.emplace_back(std::make_unique<seas_domain_writer_t>(
                cfg.domain_output->prefix, cfg.domain_output->make_adaptive_output_interval(), mesh,
                cl, seasop, PolynomialDegree));
        }
    }
    auto monitor = std::make_unique<seas_monitor_t>(seasop, std::move(writers), ts.fsal());
    ts.set_monitor(*monitor);

    const auto reduce_number = [&topo](std::size_t number) {
        std::size_t number_global;
        MPI_Reduce(&number, &number_global, 1, mpi_type_t<std::size_t>(), MPI_SUM, 0, topo->comm());
        return number_global;
    };
    std::size_t num_dofs_domain = reduce_number(seasop->domain().number_of_local_dofs());
    std::size_t num_dofs_fault = reduce_number(seasop->friction().number_of_local_dofs());

    if (rank == 0) {
        std::cout << "DOFs (domain): " << num_dofs_domain << std::endl;
        std::cout << "DOFs (fault): " << num_dofs_fault << std::endl;
    }

    Stopwatch sw;
    sw.start();
    ts.solve(cfg.final_time);
    double time = sw.stop();

    if (rank == 0) {
        auto date_time = std::time(nullptr);
        std::cout << "========= Summary =========" << std::endl;
        std::cout << "date_time=" << std::ctime(&date_time);
        std::cout << "code_version=" << VersionString << std::endl;
        std::cout << "solve_time=" << time << std::endl;
        std::cout << "time_steps=" << ts.get_step_number() << std::endl;
        std::cout << "step_rejections=" << ts.get_step_rejections() << std::endl;
        std::cout << "min_time_step=" << monitor->min_time_step() << std::endl;
        std::cout << "max_time_step=" << monitor->max_time_step() << std::endl;
        std::cout << "dofs_domain=" << num_dofs_domain << std::endl;
        std::cout << "dofs_fault=" << num_dofs_fault << std::endl;
    }

    auto solution = scenario.solution(cfg.final_time);
    if (solution) {
        seasop->update_internal_state(cfg.final_time, ts.state(), true, false, true);
        auto numeric = seasop->displacement();
        double error =
            tndm::Error<DomainDimension>::L2(*cl, numeric, *solution, 0, PETSC_COMM_WORLD);
        if (rank == 0) {
            std::cout << "L2_error_domain=" << error << std::endl;
        }
    }
    auto fault_solution = friction_scenario.solution(cfg.final_time);
    if (fault_solution) {
        auto numeric = seasop->friction().raw_state(ts.state());
        double error = tndm::Error<DomainDimension>::L2(
            mesh, *cl, numeric, seasop->friction().fault_map().localFctNos(), *fault_solution, 0,
            PETSC_COMM_WORLD);
        if (rank == 0) {
            std::cout << "L2_error_fault=" << error << std::endl;
        }
    }
    if (rank == 0) {
        std::cout << "===========================" << std::endl;
    }
}

} // namespace tndm::detail

namespace tndm {

void solveSEASProblem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    if (cfg.type == SeasType::Poisson) {
        if (cfg.discrete_green) {
            detail::solve_seas_problem<SeasType::Poisson, true>(mesh, cfg);
        } else {
            detail::solve_seas_problem<SeasType::Poisson, false>(mesh, cfg);
        }
    } else if (cfg.type == SeasType::Elasticity) {
        if (cfg.discrete_green) {
            detail::solve_seas_problem<SeasType::Elasticity, true>(mesh, cfg);
        } else {
            detail::solve_seas_problem<SeasType::Elasticity, false>(mesh, cfg);
        }
    }
}

} // namespace tndm
