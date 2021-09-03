#include "SEAS.h"
#include "common/PetscTimeSolver.h"
#include "config.h"
#include "form/BoundaryMap.h"
#include "localoperator/DieterichRuinaAgeing.h"
#include "localoperator/Elasticity.h"
#include "localoperator/Poisson.h"
#include "localoperator/RateAndState.h"
#include "tandem/DiscreteGreenAdapter.h"
#include "tandem/FrictionConfig.h"
#include "tandem/SeasElasticityAdapter.h"
#include "tandem/SeasOperator.h"
#include "tandem/SeasPoissonAdapter.h"
#include "tandem/SeasScenario.h"
#include "tandem/SeasWriter.h"

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

template <SeasType Type> struct adapter;
template <> struct adapter<SeasType::Poisson> {
    using type = SeasPoissonAdapter;
    static auto make(Config const& cfg, SeasScenario<Poisson> const& scenario,
                     std::unique_ptr<BoundaryMap> fault_map,
                     std::shared_ptr<Curvilinear<DomainDimension>> cl,
                     std::shared_ptr<DGOperatorTopo> topo,
                     std::unique_ptr<RefElement<DomainDimension - 1u>> space) {
        auto lop = std::make_unique<Poisson>(std::move(cl), scenario.mu(), DGMethod::IP);
        auto dgop = std::make_unique<DGOperator<Poisson>>(std::move(topo), std::move(lop));
        return std::make_unique<SeasPoissonAdapter>(
            std::move(fault_map), std::move(space), std::move(dgop), cfg.up, cfg.ref_normal,
            cfg.matrix_free, MGConfig(cfg.mg_coarse_level, cfg.mg_strategy));
    }
};
template <> struct adapter<SeasType::Elasticity> {
    using type = SeasElasticityAdapter;
    static auto make(Config const& cfg, SeasScenario<Elasticity> const& scenario,
                     std::unique_ptr<BoundaryMap> fault_map,
                     std::shared_ptr<Curvilinear<DomainDimension>> cl,
                     std::shared_ptr<DGOperatorTopo> topo,
                     std::unique_ptr<RefElement<DomainDimension - 1u>> space) {
        auto lop = std::make_unique<Elasticity>(std::move(cl), scenario.lam(), scenario.mu(),
                                                DGMethod::IP);
        auto dgop = std::make_unique<DGOperator<Elasticity>>(std::move(topo), std::move(lop));
        return std::make_unique<SeasElasticityAdapter>(
            std::move(fault_map), std::move(space), std::move(dgop), cfg.up, cfg.ref_normal,
            cfg.matrix_free, MGConfig(cfg.mg_coarse_level, cfg.mg_strategy));
    }
};

template <class Adapter, bool MakeGreen> struct discrete_green;
template <class Adapter> struct discrete_green<Adapter, false> {
    using type = Adapter;
    static auto wrap(std::unique_ptr<Adapter> adapter, std::size_t) { return std::move(adapter); }
};
template <class Adapter> struct discrete_green<Adapter, true> {
    using type = DiscreteGreenAdapter<Adapter>;
    static auto wrap(std::unique_ptr<Adapter> adapter, std::size_t slip_block_size) {
        return std::make_unique<type>(std::move(adapter), slip_block_size);
    }
};

template <SeasType Type, bool MakeGreen>
void solve_seas_problem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    using base_adapter_t = typename adapter<Type>::type;
    using adapter_lop_t = typename adapter<Type>::type::local_operator_t;
    using adapter_t = typename discrete_green<base_adapter_t, MakeGreen>::type;
    using fault_op_t = RateAndState<DieterichRuinaAgeing>;
    using seas_op_t = SeasOperator<fault_op_t, adapter_t>;
    using seas_fault_probe_writer_t = SeasFaultProbeWriter<DomainDimension, seas_op_t>;
    using seas_domain_probe_writer_t = SeasDomainProbeWriter<DomainDimension, seas_op_t>;
    using seas_fault_writer_t = SeasFaultWriter<DomainDimension, seas_op_t>;
    using seas_domain_writer_t = SeasDomainWriter<DomainDimension, seas_op_t>;
    using seas_monitor_t = SeasMonitor<seas_op_t>;

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    auto scenario = SeasScenario<adapter_lop_t>(cfg.lib, cfg.scenario);
    auto friction_scenario = DieterichRuinaAgeingScenario(cfg.lib, cfg.scenario);

    auto cl = std::make_shared<Curvilinear<DomainDimension>>(mesh, scenario.transform(),
                                                             PolynomialDegree);

    auto fop = std::make_unique<fault_op_t>(cl);
    auto topo = std::make_shared<DGOperatorTopo>(mesh, PETSC_COMM_WORLD);
    auto fault_map = std::make_unique<BoundaryMap>(mesh, BC::Fault, PETSC_COMM_WORLD);
    auto adapt = discrete_green<base_adapter_t, MakeGreen>::wrap(
        adapter<Type>::make(cfg, scenario, std::move(fault_map), cl, topo, fop->space().clone()),
        fop->slip_block_size());
    adapt->warmup();

    auto seasop = std::make_shared<seas_op_t>(std::move(fop), std::move(adapt));
    seasop->lop().set_constant_params(friction_scenario.constant_params());
    seasop->lop().set_params(friction_scenario.param_fun());
    if (friction_scenario.source_fun()) {
        seasop->lop().set_source_fun(*friction_scenario.source_fun());
    }
    if (scenario.boundary()) {
        seasop->set_boundary(*scenario.boundary());
    }

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
            writers.emplace_back(std::make_unique<SeasFaultScalarWriter<seas_op_t>>(
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
    std::size_t num_dofs_domain = reduce_number(seasop->adapter().number_of_local_dofs());
    std::size_t num_dofs_fault = reduce_number(seasop->number_of_local_dofs());

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
        auto numeric = seasop->adapter().displacement();
        double error =
            tndm::Error<DomainDimension>::L2(*cl, numeric, *solution, 0, PETSC_COMM_WORLD);
        if (rank == 0) {
            std::cout << "L2_error_domain=" << error << std::endl;
        }
    }
    auto fault_solution = friction_scenario.solution(cfg.final_time);
    if (fault_solution) {
        auto numeric = seasop->raw_state(ts.state());
        double error =
            tndm::Error<DomainDimension>::L2(mesh, *cl, numeric, seasop->faultMap().localFctNos(),
                                             *fault_solution, 0, PETSC_COMM_WORLD);
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
