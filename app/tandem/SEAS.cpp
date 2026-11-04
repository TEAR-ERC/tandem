#include "SEAS.h"
#include "common/PetscTimeSolver.h"
#include "config.h"
#include "form/AbstractDGOperator.h"
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
#include "localoperator/DieterichRuinaSlip.h"
#include "localoperator/Elasticity.h"
#include "localoperator/Poisson.h"
#include "localoperator/RateAndState.h"
#include "mesh/LocalSimplexMesh.h"
#include "tandem/Context.h"
#include "tandem/ContextBase.h"
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
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace tndm::detail {

template <typename Type>
auto make_context(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    return std::make_unique<seas::Context<Type>>(
        mesh, std::make_unique<SeasScenario<Type>>(cfg.lib, cfg.scenario),
        std::make_unique<DieterichRuinaScenario>(cfg.lib, cfg.scenario), cfg.up, cfg.ref_normal);
}

template <std::size_t N>
auto make_state_vecs(std::array<std::size_t, N> const& block_sizes,
                     std::array<std::size_t, N> const& num_elements, MPI_Comm comm) {
    std::array<std::unique_ptr<PetscVector>, N> state;
    for (std::size_t n = 0; n < N; ++n) {
        state[n] = std::make_unique<PetscVector>(block_sizes[n], num_elements[n], comm);
    }
    return state;
}

/// State of an output writer as stored in last_checkpoint.txt: output step, time and VMax of the
/// last output
using WriterState = std::tuple<std::size_t, double, double>;
using WriterStates = std::map<std::string, WriterState>;

/**
 * @brief Read the writer states from last_checkpoint.txt
 *
 * Lines have the form "<writer key> <step> <time> <VMax>"; the first line (checkpoint path) and
 * lines starting with '#' are skipped. Returns an empty map if load_path is not a regular file
 * (e.g. if the checkpoint directory itself was given) or if the file holds no writer states, in
 * which case the writers start from scratch as before.
 */
WriterStates read_writer_states(std::string const& load_path, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Rank 0 reads the file, all ranks parse the same content
    std::string content;
    int ok = 1;
    if (rank == 0) {
        if (std::filesystem::is_regular_file(load_path)) {
            std::ifstream file(load_path);
            std::stringstream ss;
            ss << file.rdbuf();
            content = ss.str();
            std::cout << "Retrieving writer states from " << load_path << std::endl;
        } else {
            ok = 0;
        }
    }
    int size = static_cast<int>(content.size());
    MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    if (!ok) {
        return {};
    }
    MPI_Bcast(&size, 1, MPI_INT, 0, comm);
    content.resize(size);
    MPI_Bcast(content.data(), size, MPI_CHAR, 0, comm);

    WriterStates states;
    std::istringstream in(content);
    std::string line;
    std::getline(in, line); // checkpoint path
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ls(line);
        std::string key;
        std::size_t step;
        double time, VMax;
        if (!(ls >> key >> step >> time >> VMax)) {
            throw std::runtime_error("Malformed writer state in " + load_path + ": " + line);
        }
        states[key] = WriterState{step, time, VMax};
        if (rank == 0) {
            std::cout << "Loaded " << key << " state: step=" << step << ", time=" << time
                      << ", VMax=" << VMax << std::endl;
        }
    }
    return states;
}

/// Writes "<key> <step> <time> <VMax>" for every writer that takes part in checkpointing.
/// Called by the checkpoint code on rank 0 (writer states are identical on all ranks).
PetscErrorCode write_writer_states(PetscViewer viewer, void* ctx) {
    auto const& monitor = *static_cast<seas::Monitor const*>(ctx);
    for (auto const& writer : monitor.writers()) {
        if (writer->checkpoint_key().empty()) {
            continue;
        }
        // %.17g round-trips doubles exactly (VMax is often ~1e-9 and must not be rounded)
        PetscErrorCode ierr = PetscViewerASCIIPrintf(
            viewer, "%s %zu %.17g %.17g\n", writer->checkpoint_key().c_str(), writer->output_step(),
            writer->last_output_time(), writer->last_output_VMax());
        CHKERRQ(ierr);
    }
    return 0;
}

auto add_writers(Config const& cfg, LocalSimplexMesh<DomainDimension> const& mesh,
                 std::shared_ptr<Curvilinear<DomainDimension>> cl, BoundaryMap const& fault_map,
                 seas::Monitor& monitor, MPI_Comm comm) {
#ifdef ENABLE_HDF5
    bool const checkpoint_enabled =
        cfg.ts_checkpoint_config.storage_type != TsCheckpointStorageType::NONE;
#endif
    if (cfg.fault_output && cfg.domain_output) {
        if (cfg.fault_output->prefix == cfg.domain_output->prefix) {
            throw std::runtime_error(
                "Fault output prefix and domain output prefix must not be identical");
        }
    }

    WriterStates writer_states;
    if (cfg.ts_checkpoint_config.load_directory.has_value()) {
        writer_states = read_writer_states(*cfg.ts_checkpoint_config.load_directory, comm);
    }
    // Name the writer for checkpoint bookkeeping, restore its state after a restart (this also
    // truncates its files back to the checkpoint), and hand it over to the monitor.
    auto register_writer = [&](std::unique_ptr<seas::Writer> writer, std::string key) {
        writer->set_checkpoint_key(key);
        if (auto it = writer_states.find(key); it != writer_states.end()) {
            auto const& [step, time, VMax] = it->second;
            writer->set_state(step, time, VMax);
        }
        monitor.add_writer(std::move(writer));
    };

    if (cfg.fault_probe_output && !cfg.fault_probe_output->probes.empty()) {
        auto const& oc = *cfg.fault_probe_output;
        std::unique_ptr<seas::Writer> writer;
        switch (oc.type) {
#ifdef ENABLE_HDF5
        case TableWriterType::HDF5:
            writer = std::make_unique<seas::HDF5CommonProbeWriter<DomainDimension, true>>(
                oc.prefix, oc.probes, oc.make_adaptive_output_interval(), mesh, cl, fault_map, comm,
                checkpoint_enabled);
            break;
#endif
        default:
            // TableWriterType::HDF5 is rejected at parse time in non-HDF5 builds (SeasConfig.cpp)
            // so this default handles only CSV and Tecplot.
            writer = std::make_unique<seas::FaultProbeWriter<DomainDimension>>(
                oc.prefix, oc.make_writer(), oc.probes, oc.make_adaptive_output_interval(), mesh,
                cl, fault_map, comm);
            break;
        }

        register_writer(std::move(writer), "fault_probe_output");
    }

    if (cfg.domain_probe_output && !cfg.domain_probe_output->probes.empty()) {
        auto const& oc = *cfg.domain_probe_output;
        std::unique_ptr<seas::Writer> writer;
        switch (oc.type) {
#ifdef ENABLE_HDF5
        case TableWriterType::HDF5:
            writer = std::make_unique<seas::HDF5CommonProbeWriter<DomainDimension, false>>(
                oc.prefix, oc.probes, oc.make_adaptive_output_interval(), mesh, cl, fault_map, comm,
                checkpoint_enabled);
            break;
#endif
        default:
            // TableWriterType::HDF5 is rejected at parse time in non-HDF5 builds (SeasConfig.cpp)
            // so this default handles only CSV and Tecplot.
            writer = std::make_unique<seas::DomainProbeWriter<DomainDimension>>(
                oc.prefix, oc.make_writer(), oc.probes, oc.make_adaptive_output_interval(), mesh,
                cl, comm);
            break;
        }

        register_writer(std::move(writer), "domain_probe_output");
    }

    if (cfg.fault_output) {
        auto const& oc = *cfg.fault_output;
        auto writer = std::make_unique<seas::FaultWriter<DomainDimension>>(
            oc.prefix, oc.make_adaptive_output_interval(), mesh, cl, PolynomialDegree, fault_map,
            comm);
        register_writer(std::move(writer), "fault_output");
    }

    if (cfg.fault_scalar_output) {
        auto const& oc = *cfg.fault_scalar_output;
        auto writer = std::make_unique<seas::FaultScalarWriter>(
            oc.prefix, oc.make_writer(), oc.make_adaptive_output_interval(), comm);
        register_writer(std::move(writer), "fault_scalar_output");
    }

    if (cfg.domain_output) {
        auto const& oc = *cfg.domain_output;
        auto writer = std::make_unique<seas::DomainWriter<DomainDimension>>(
            oc.prefix, oc.make_adaptive_output_interval(), mesh, cl, PolynomialDegree, oc.jacobian,
            comm);
        register_writer(std::move(writer), "domain_output");
    }
#ifdef ENABLE_HDF5
    if (cfg.moment_rate_output) {
        auto const& oc = *cfg.moment_rate_output;
        monitor.add_writer(std::make_unique<seas::MomentRateWriter<DomainDimension>>(
            oc.prefix, oc.make_adaptive_output_interval(), mesh, cl, PolynomialDegree, fault_map,
            comm, checkpoint_enabled));
    }
#else
    if (cfg.moment_rate_output) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) {
            std::cerr << "Warning: moment_rate_output requires HDF5 support. "
                         "Reconfigure with -DENABLE_HDF5=ON to enable this output.\n";
        }
    }
#endif
}

template <typename seas_t> struct operator_specifics;

template <typename T> struct qd_operator_specifics {
    using monitor_t = seas::MonitorQD;

    static auto make(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg,
                     seas::ContextBase& ctx) {
        auto seasop = std::make_shared<T>(std::move(ctx.dg()), std::move(ctx.adapter()),
                                          std::move(ctx.friction()), cfg.matrix_free,
                                          MGConfig(cfg.mg_coarse_level, cfg.mg_strategy));
        ctx.setup_seasop(*seasop);
        seasop->warmup();
        return seasop;
    }

    static std::optional<double> cfl_time_step(T const&) { return std::nullopt; }

    template <typename TimeSolver>
    static auto displacement(double time, TimeSolver const& ts, T& seasop) {
        seasop.update_internal_state(time, ts.state(0), true, false, true);
        return seasop.displacement();
    }

    template <typename TimeSolver> static auto state(double, TimeSolver const& ts, T& seasop) {
        return seasop.friction().raw_state(ts.state(0));
    }

    static void print_profile(T const&) {}
};

template <>
struct operator_specifics<SeasQDOperator> : public qd_operator_specifics<SeasQDOperator> {};

template <>
struct operator_specifics<SeasQDDiscreteGreenOperator>
    : public qd_operator_specifics<SeasQDDiscreteGreenOperator> {

    static auto make(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg,
                     seas::ContextBase& ctx) {
        auto const& cfgcp = cfg.gf_checkpoint_config;

        std::optional<std::string> prefix;
        double freq_cputime;
        if (!cfgcp) {
            prefix = std::nullopt;
            freq_cputime = 1e10;
        } else {
            prefix = cfgcp->prefix;
            freq_cputime = cfgcp->frequency_cputime_minutes;
        }

        auto seasop = std::make_shared<SeasQDDiscreteGreenOperator>(
            std::move(ctx.dg()), std::move(ctx.adapter()), std::move(ctx.friction()), mesh, prefix,
            freq_cputime, cfg.matrix_free, MGConfig(cfg.mg_coarse_level, cfg.mg_strategy));
        ctx.setup_seasop(*seasop);
        seasop->warmup();
        return seasop;
    }
};

template <> struct operator_specifics<SeasFDOperator> {
    using monitor_t = seas::MonitorFD;

    static auto make(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg,
                     seas::ContextBase& ctx) {
        auto seasop = std::make_shared<SeasFDOperator>(
            std::move(ctx.dg()), std::move(ctx.adapter()), std::move(ctx.friction()));
        ctx.setup_seasop(*seasop);
        return seasop;
    }

    static std::optional<double> cfl_time_step(SeasFDOperator const& seasop) {
        return std::make_optional(seasop.cfl_time_step());
    }

    template <typename TimeSolver>
    static auto displacement(double, TimeSolver const& ts, SeasFDOperator& seasop) {
        return seasop.domain_function(ts.state(1));
    }

    template <typename TimeSolver>
    static auto state(double, TimeSolver const& ts, SeasFDOperator& seasop) {
        return seasop.friction().raw_state(ts.state(2));
    }

    static void print_profile(SeasFDOperator const& seasop) {
        seasop.profile().print(std::cout, seasop.comm());
    }
};

template <typename seas_t>
void solve_seas_problem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg,
                        seas::ContextBase& ctx) {
    auto seasop = operator_specifics<seas_t>::make(mesh, cfg, ctx);

    auto ts = PetscTimeSolver(
        *seasop,
        make_state_vecs(seasop->block_sizes(), seasop->num_local_elements(), seasop->comm()), cfg);

    auto cfl_time_step = operator_specifics<seas_t>::cfl_time_step(*seasop);
    if (cfl_time_step) {
        ts.set_max_time_step(*cfl_time_step * cfg.cfl);
    }

    auto monitor =
        std::make_unique<typename operator_specifics<seas_t>::monitor_t>(seasop, ts.fsal());
    add_writers(cfg, mesh, ctx.cl, seasop->adapter().fault_map(), *monitor, seasop->comm());
    monitor->write_static();

    ts.set_monitor(*monitor);
    ts.set_checkpoint_info_callback(write_writer_states, monitor.get());

    int rank;
    MPI_Comm comm = seasop->comm();
    MPI_Comm_rank(comm, &rank);

    const auto reduce_number = [&comm](std::size_t number) {
        std::size_t number_global;
        MPI_Reduce(&number, &number_global, 1, mpi_type_t<std::size_t>(), MPI_SUM, 0, comm);
        return number_global;
    };
    std::size_t num_dofs_domain = reduce_number(seasop->domain().number_of_local_dofs());
    std::size_t num_dofs_fault = reduce_number(seasop->friction().number_of_local_dofs());

    double local_mesh_size = ctx.cl->local_mesh_size();
    double mesh_size;
    MPI_Reduce(&local_mesh_size, &mesh_size, 1, mpi_type_t<double>(), MPI_MAX, 0, comm);

    if (rank == 0) {
        std::cout << "DOFs (domain): " << num_dofs_domain << std::endl;
        std::cout << "DOFs (fault): " << num_dofs_fault << std::endl;
        std::cout << "Mesh size: " << mesh_size << std::endl;
        if (cfl_time_step) {
            std::cout << "CFL time step: " << *cfl_time_step << std::endl;
        }
    }

    Stopwatch sw;
    sw.start();
    ts.solve(cfg.final_time);

    double solve_time = sw.stop();

    std::optional<double> L2_error_domain = std::nullopt;
    auto domain_solution = ctx.domain_solution(cfg.final_time);
    if (domain_solution) {
        auto numeric = operator_specifics<seas_t>::displacement(cfg.final_time, ts, *seasop);
        L2_error_domain =
            tndm::Error<DomainDimension>::L2(*ctx.cl, numeric, *domain_solution, 0, seasop->comm());
    }
    std::optional<double> L2_error_fault = std::nullopt;
    auto fault_solution = ctx.fault_solution(cfg.final_time);
    if (fault_solution) {
        auto numeric = operator_specifics<seas_t>::state(cfg.final_time, ts, *seasop);
        L2_error_fault = tndm::Error<DomainDimension>::L2(
            mesh, *ctx.cl, numeric, seasop->adapter().fault_map().localFctNos(), *fault_solution, 0,
            seasop->comm());
    }

    if (rank == 0) {
        std::cout << std::endl;
    }
    operator_specifics<seas_t>::print_profile(*seasop);
    if (rank == 0) {
        std::cout << std::endl;
    }

    if (rank == 0) {
        auto date_time = std::time(nullptr);
        std::cout << "========= Summary =========" << std::endl;
        std::cout << "date_time=" << std::ctime(&date_time);
        std::cout << "code_version=" << VersionString << std::endl;
        std::cout << "solve_time=" << solve_time << std::endl;
        std::cout << "time_steps=" << ts.get_step_number() << std::endl;
        std::cout << "step_rejections=" << ts.get_step_rejections() << std::endl;
        std::cout << "min_time_step=" << monitor->min_time_step() << std::endl;
        std::cout << "max_time_step=" << monitor->max_time_step() << std::endl;
        std::cout << "dofs_domain=" << num_dofs_domain << std::endl;
        std::cout << "dofs_fault=" << num_dofs_fault << std::endl;
        std::cout << "mesh_size=" << mesh_size << std::endl;
        if (cfl_time_step) {
            std::cout << "dt_cfl=" << *cfl_time_step << std::endl;
        }
        if (L2_error_domain) {
            std::cout << "L2_error_domain=" << *L2_error_domain << std::endl;
        }
        if (L2_error_fault) {
            std::cout << "L2_error_fault=" << *L2_error_fault << std::endl;
        }
        std::cout << "===========================" << std::endl;
    }
}

} // namespace tndm::detail

namespace tndm {

void solveSEASProblem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    std::unique_ptr<seas::ContextBase> ctx = nullptr;
    switch (cfg.type) {
    case LocalOpType::Poisson:
        ctx = detail::make_context<Poisson>(mesh, cfg);
        break;
    case LocalOpType::Elasticity:
        ctx = detail::make_context<Elasticity>(mesh, cfg);
        break;
    default:
        throw std::runtime_error("Unknown seas type");
        break;
    };
    switch (cfg.mode) {
    case SeasMode::QuasiDynamicDiscreteGreen:
        detail::solve_seas_problem<SeasQDDiscreteGreenOperator>(mesh, cfg, *ctx);
        break;
    case SeasMode::QuasiDynamic:
        detail::solve_seas_problem<SeasQDOperator>(mesh, cfg, *ctx);
        break;
    case SeasMode::FullyDynamic:
        detail::solve_seas_problem<SeasFDOperator>(mesh, cfg, *ctx);
        break;
    default:
        break;
    }
}

} // namespace tndm
