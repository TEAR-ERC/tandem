#include "SEAS.h"
#include "common/PetscTimeSolver.h"
#include "config.h"
#include "form/AdapterOperator.h"
#include "form/BoundaryMap.h"
#include "form/FacetFunctionalFactory.h"
#include "form/FiniteElementFunction.h"
#include "form/FrictionOperator.h"
#include "form/SeasFDOperator.h"
#include "form/SeasQDDiscreteGreenOperator.h"
#include "form/SeasQDOperator.h"
#include "form/VolumeFunctionalFactory.h"
#include "localoperator/Adapter.h"
#include "localoperator/DieterichRuinaAgeing.h"
#include "localoperator/Elasticity.h"
#include "localoperator/Poisson.h"
#include "localoperator/RateAndState.h"
#include "mesh/LocalSimplexMesh.h"
#include "tandem/FrictionConfig.h"
#include "tandem/Monitor.h"
#include "tandem/SeasScenario.h"
#include "tandem/Writer.h"

#include "form/DGOperator.h"
#include "form/DGOperatorTopo.h"
#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "parallel/MPITraits.h"
#include "tensor/Managed.h"
#include "util/Stopwatch.h"

#include <limits>
#include <mpi.h>
#include <petscsys.h>

#include <algorithm>
#include <array>
#include <ctime>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace tndm::detail {

template <typename Type> struct make_lop;
template <> struct make_lop<Poisson> {
    static auto dg(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                   SeasScenario<Poisson> const& scenario) {
        return std::make_shared<Poisson>(std::move(cl), scenario.mu(), DGMethod::IP);
    }
};
template <> struct make_lop<Elasticity> {
    static auto dg(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                   SeasScenario<Elasticity> const& scenario) {
        return std::make_shared<Elasticity>(std::move(cl), scenario.lam(), scenario.mu(),
                                            DGMethod::IP);
    }
};

template <typename Type> struct make_op {
    using adapter_t = AdapterOperator<Type>;
    using dg_t = DGOperator<Type>;
    using friction_lop_t = RateAndState<DieterichRuinaAgeing>;
    using friction_t = FrictionOperator<friction_lop_t>;

    make_op(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg)
        : scenario(cfg.lib, cfg.scenario),
          friction_scenario(DieterichRuinaAgeingScenario(cfg.lib, cfg.scenario)) {

        cl = std::make_shared<Curvilinear<DomainDimension>>(mesh, scenario.transform(),
                                                            PolynomialDegree);
        fault_map = std::make_shared<BoundaryMap>(mesh, BC::Fault, PETSC_COMM_WORLD);
        topo = std::make_shared<DGOperatorTopo>(mesh, PETSC_COMM_WORLD);
        dg_lop = make_lop<Type>::dg(cl, scenario);
    }

    auto dg() { return std::make_unique<dg_t>(topo, dg_lop); }
    auto friction() {
        auto fric =
            std::make_unique<friction_t>(std::make_unique<friction_lop_t>(cl), topo, fault_map);
        fric->lop().set_constant_params(friction_scenario.constant_params());
        fric->lop().set_params(friction_scenario.param_fun());
        if (friction_scenario.source_fun()) {
            fric->lop().set_source_fun(*friction_scenario.source_fun());
        }
        return fric;
    }
    auto adapter(Config const& cfg, dg_t& dg, friction_t& friction) {
        return std::make_unique<adapter_t>(
            dg_lop,
            std::make_unique<Adapter<Type>>(cl, friction.lop().space().clone(),
                                            dg_lop->facetQuadratureRule(), cfg.up, cfg.ref_normal),
            topo, fault_map);
    }

    SeasScenario<Type> scenario;
    DieterichRuinaAgeingScenario friction_scenario;
    std::shared_ptr<Curvilinear<DomainDimension>> cl;
    std::shared_ptr<BoundaryMap> fault_map;
    std::shared_ptr<DGOperatorTopo> topo;
    std::shared_ptr<Type> dg_lop;
};

template <std::size_t N>
auto make_state_vecs(std::array<std::size_t, N> const& block_sizes,
                     std::array<std::size_t, N> const& num_elements, MPI_Comm comm) {
    std::array<std::unique_ptr<PetscVector>, N> state;
    for (std::size_t n = 0; n < N; ++n) {
        state[n] = std::make_unique<PetscVector>(block_sizes[n], num_elements[n], comm);
    }
    return state;
}

auto add_writers(Config const& cfg, LocalSimplexMesh<DomainDimension> const& mesh,
                 std::shared_ptr<Curvilinear<DomainDimension>> cl, BoundaryMap const& fault_map,
                 seas::Monitor& monitor, MPI_Comm comm) {
    if (cfg.fault_output && cfg.domain_output) {
        if (cfg.fault_output->prefix == cfg.domain_output->prefix) {
            throw std::runtime_error(
                "Fault output prefix and domain output prefix must not be identical");
        }
    }
    if (cfg.fault_probe_output) {
        monitor.add_writer(std::make_unique<seas::FaultProbeWriter<DomainDimension>>(
            cfg.fault_probe_output->prefix, cfg.fault_probe_output->probes,
            cfg.fault_probe_output->make_adaptive_output_interval(), mesh, cl, fault_map, comm));
    }
    if (cfg.domain_probe_output) {
        monitor.add_writer(std::make_unique<seas::DomainProbeWriter<DomainDimension>>(
            cfg.domain_probe_output->prefix, cfg.domain_probe_output->probes,
            cfg.domain_probe_output->make_adaptive_output_interval(), mesh, cl, comm));
    }
    if (cfg.fault_output) {
        monitor.add_writer(std::make_unique<seas::FaultWriter<DomainDimension>>(
            cfg.fault_output->prefix, cfg.fault_output->make_adaptive_output_interval(), mesh, cl,
            PolynomialDegree, fault_map, comm));
    }
    if (cfg.fault_scalar_output) {
        monitor.add_writer(std::make_unique<seas::FaultScalarWriter>(
            cfg.fault_scalar_output->prefix,
            cfg.fault_scalar_output->make_adaptive_output_interval(), comm));
    }
    if (cfg.domain_output) {
        monitor.add_writer(std::make_unique<seas::DomainWriter<DomainDimension>>(
            cfg.domain_output->prefix, cfg.domain_output->make_adaptive_output_interval(), mesh, cl,
            PolynomialDegree, comm));
    }
}

auto compute_and_print_dofs(std::size_t num_local_dofs_domain, std::size_t num_local_dofs_fault,
                            MPI_Comm comm) -> std::array<std::size_t, 2> {
    int rank;
    MPI_Comm_rank(comm, &rank);

    const auto reduce_number = [&comm](std::size_t number) {
        std::size_t number_global;
        MPI_Reduce(&number, &number_global, 1, mpi_type_t<std::size_t>(), MPI_SUM, 0, comm);
        return number_global;
    };
    std::size_t num_dofs_domain = reduce_number(num_local_dofs_domain);
    std::size_t num_dofs_fault = reduce_number(num_local_dofs_fault);

    if (rank == 0) {
        std::cout << "DOFs (domain): " << num_dofs_domain << std::endl;
        std::cout << "DOFs (fault): " << num_dofs_fault << std::endl;
    }

    return {num_dofs_domain, num_dofs_fault};
}

auto print_summary(double solve_time, PetscTimeSolverBase const& ts, seas::Monitor const& monitor,
                   std::size_t num_dofs_domain, std::size_t num_dofs_fault,
                   std::optional<double> L2_error_domain, std::optional<double> L2_error_fault,
                   MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        auto date_time = std::time(nullptr);
        std::cout << "========= Summary =========" << std::endl;
        std::cout << "date_time=" << std::ctime(&date_time);
        std::cout << "code_version=" << VersionString << std::endl;
        std::cout << "solve_time=" << solve_time << std::endl;
        std::cout << "time_steps=" << ts.get_step_number() << std::endl;
        std::cout << "step_rejections=" << ts.get_step_rejections() << std::endl;
        std::cout << "min_time_step=" << monitor.min_time_step() << std::endl;
        std::cout << "max_time_step=" << monitor.max_time_step() << std::endl;
        std::cout << "dofs_domain=" << num_dofs_domain << std::endl;
        std::cout << "dofs_fault=" << num_dofs_fault << std::endl;
        if (L2_error_domain) {
            std::cout << "L2_error_domain=" << *L2_error_domain << std::endl;
        }
        if (L2_error_fault) {
            std::cout << "L2_error_fault=" << *L2_error_fault << std::endl;
        }
        std::cout << "===========================" << std::endl;
    }
}

auto L2_error_domain(Curvilinear<DomainDimension> const& cl,
                     std::function<FiniteElementFunction<DomainDimension>()> compute_numeric,
                     std::unique_ptr<SolutionInterface> solution, MPI_Comm comm)
    -> std::optional<double> {
    std::optional<double> error = std::nullopt;
    if (solution) {
        auto numeric = compute_numeric();
        error = tndm::Error<DomainDimension>::L2(cl, numeric, *solution, 0, comm);
    }
    return error;
}

auto L2_error_fault(LocalSimplexMesh<DomainDimension> const& mesh,
                    Curvilinear<DomainDimension> const& cl,
                    std::function<FiniteElementFunction<DomainDimension - 1u>()> compute_numeric,
                    BoundaryMap const& fault_map, std::unique_ptr<SolutionInterface> solution,
                    MPI_Comm comm) -> std::optional<double> {
    std::optional<double> error = std::nullopt;
    if (solution) {
        auto numeric = compute_numeric();
        error = tndm::Error<DomainDimension>::L2(mesh, cl, numeric, fault_map.localFctNos(),
                                                 *solution, 0, comm);
    }
    return error;
}

template <typename Type, bool MakeGreen>
void solve_seas_qd_problem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    using seas_t = std::conditional_t<MakeGreen, SeasQDDiscreteGreenOperator, SeasQDOperator>;
    using seas_monitor_t = seas::MonitorQD<seas_t>;

    auto op = make_op<Type>(mesh, cfg);

    auto dgop = op.dg();
    auto friction = op.friction();
    auto adapter = op.adapter(cfg, *dgop, *friction);

    auto seasop =
        std::make_shared<seas_t>(std::move(dgop), std::move(adapter), std::move(friction),
                                 cfg.matrix_free, MGConfig(cfg.mg_coarse_level, cfg.mg_strategy));
    if (op.scenario.boundary()) {
        seasop->set_boundary(std::make_unique<FacetFunctionalFactory<Type>>(
            op.dg_lop, *op.scenario.boundary(), cfg.ref_normal));
    }
    seasop->warmup();

    auto ts =
        PetscTimeSolver(*seasop, make_state_vecs(seasop->block_sizes(),
                                                 seasop->num_local_elements(), seasop->comm()));

    auto monitor = std::make_unique<seas_monitor_t>(seasop, ts.fsal());
    add_writers(cfg, mesh, op.cl, seasop->adapter().fault_map(), *monitor, seasop->comm());
    ts.set_monitor(*monitor);

    auto [num_dofs_domain, num_dofs_fault] =
        compute_and_print_dofs(seasop->domain().number_of_local_dofs(),
                               seasop->friction().number_of_local_dofs(), seasop->comm());

    Stopwatch sw;
    sw.start();
    ts.solve(cfg.final_time);
    double time = sw.stop();

    auto L2_domain = L2_error_domain(
        *op.cl,
        [&]() {
            seasop->update_internal_state(cfg.final_time, ts.state(0), true, false, true);
            return seasop->displacement();
        },
        op.scenario.solution(cfg.final_time), seasop->comm());

    auto L2_fault = L2_error_fault(
        mesh, *op.cl, [&]() { return seasop->friction().raw_state(ts.state(0)); },
        seasop->adapter().fault_map(), op.friction_scenario.solution(cfg.final_time),
        seasop->comm());

    print_summary(time, ts, *monitor, num_dofs_domain, num_dofs_fault, L2_domain, L2_fault,
                  seasop->comm());
}

template <typename Type>
void solve_seas_fd_problem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    using seas_t = SeasFDOperator;
    using seas_monitor_t = seas::MonitorFD<seas_t>;

    auto op = make_op<Type>(mesh, cfg);

    auto dgop = op.dg();
    auto friction = op.friction();
    auto adapter = op.adapter(cfg, *dgop, *friction);

    auto seasop =
        std::make_shared<seas_t>(std::move(dgop), std::move(adapter), std::move(friction));
    if (op.scenario.boundary()) {
        seasop->set_boundary(std::make_unique<FacetFunctionalFactory<Type>>(
            op.dg_lop, *op.scenario.boundary(), cfg.ref_normal));
    }
    if (op.scenario.initial_displacement()) {
        seasop->set_initial_displacement(std::make_unique<VolumeFunctionalFactory<Type>>(
            op.dg_lop, *op.scenario.initial_displacement()));
    }
    if (op.scenario.initial_velocity()) {
        seasop->set_initial_velocity(std::make_unique<VolumeFunctionalFactory<Type>>(
            op.dg_lop, *op.scenario.initial_velocity()));
    }

    auto ts =
        PetscTimeSolver(*seasop, make_state_vecs(seasop->block_sizes(),
                                                 seasop->num_local_elements(), seasop->comm()));

    auto monitor = std::make_unique<seas_monitor_t>(seasop, ts.fsal());
    add_writers(cfg, mesh, op.cl, seasop->adapter().fault_map(), *monitor, seasop->comm());
    ts.set_monitor(*monitor);

    auto [num_dofs_domain, num_dofs_fault] =
        compute_and_print_dofs(seasop->domain().number_of_local_dofs(),
                               seasop->friction().number_of_local_dofs(), seasop->comm());

    Stopwatch sw;
    sw.start();
    ts.solve(cfg.final_time);
    double time = sw.stop();

    auto L2_domain = L2_error_domain(
        *op.cl, [&]() { return seasop->domain_function(ts.state(1)); },
        op.scenario.solution(cfg.final_time), seasop->comm());

    auto L2_fault = L2_error_fault(
        mesh, *op.cl, [&]() { return seasop->friction().raw_state(ts.state(2)); },
        seasop->adapter().fault_map(), op.friction_scenario.solution(cfg.final_time),
        seasop->comm());

    print_summary(time, ts, *monitor, num_dofs_domain, num_dofs_fault, L2_domain, L2_fault,
                  seasop->comm());
}

} // namespace tndm::detail

namespace tndm {

void solveSEASProblem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    if (cfg.type == SeasType::Poisson) {
        switch (cfg.mode) {
        case SeasMode::QuasiDynamicDiscreteGreen:
            detail::solve_seas_qd_problem<Poisson, true>(mesh, cfg);
            break;
        case SeasMode::QuasiDynamic:
            detail::solve_seas_qd_problem<Poisson, false>(mesh, cfg);
            break;
        case SeasMode::FullyDynamic:
            detail::solve_seas_fd_problem<Poisson>(mesh, cfg);
            break;
        default:
            break;
        }
    } else if (cfg.type == SeasType::Elasticity) {
        switch (cfg.mode) {
        case SeasMode::QuasiDynamicDiscreteGreen:
            detail::solve_seas_qd_problem<Elasticity, true>(mesh, cfg);
            break;
        case SeasMode::QuasiDynamic:
            detail::solve_seas_qd_problem<Elasticity, false>(mesh, cfg);
            break;
        case SeasMode::FullyDynamic:
            detail::solve_seas_fd_problem<Elasticity>(mesh, cfg);
            break;
        default:
            break;
        }
    }
}

} // namespace tndm
