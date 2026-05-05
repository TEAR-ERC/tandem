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

#include <cstdio>
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
        auto const& oc = *cfg.fault_probe_output;
        monitor.add_writer(std::make_unique<seas::FaultProbeWriter<DomainDimension>>(
            oc.prefix, oc.make_writer(), oc.probes, oc.make_adaptive_output_interval(), mesh, cl,
            fault_map, comm));
    }
    if (cfg.domain_probe_output) {
        auto const& oc = *cfg.domain_probe_output;
        monitor.add_writer(std::make_unique<seas::DomainProbeWriter<DomainDimension>>(
            oc.prefix, oc.make_writer(), oc.probes, oc.make_adaptive_output_interval(), mesh, cl,
            comm));
    }
    if (cfg.fault_output) {
        auto const& oc = *cfg.fault_output;
        monitor.add_writer(std::make_unique<seas::FaultWriter<DomainDimension>>(
            oc.prefix, oc.make_adaptive_output_interval(), mesh, cl, PolynomialDegree, fault_map,
            comm));
    }
    if (cfg.fault_scalar_output) {
        auto const& oc = *cfg.fault_scalar_output;
        monitor.add_writer(std::make_unique<seas::FaultScalarWriter>(
            oc.prefix, oc.make_writer(), oc.make_adaptive_output_interval(), comm));
    }
    if (cfg.domain_output) {
        auto const& oc = *cfg.domain_output;
        monitor.add_writer(std::make_unique<seas::DomainWriter<DomainDimension>>(
            oc.prefix, oc.make_adaptive_output_interval(), mesh, cl, PolynomialDegree, oc.jacobian,
            comm));
    }
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
            freq_cputime, cfg.matrix_free, MGConfig(cfg.mg_coarse_level, cfg.mg_strategy),
            cfg.hmatrix_config);
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

double validateGFHMatrix(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    if (cfg.mode != SeasMode::QuasiDynamicDiscreteGreen) {
        throw std::runtime_error("validateGFHMatrix requires mode = QDGreen");
    }

    Config cfg_h = cfg;
    cfg_h.hmatrix_config.use_hmatrix = true;

    std::unique_ptr<seas::ContextBase> ctx = nullptr;
    switch (cfg_h.type) {
    case LocalOpType::Poisson:
        ctx = detail::make_context<Poisson>(mesh, cfg_h);
        break;
    case LocalOpType::Elasticity:
        ctx = detail::make_context<Elasticity>(mesh, cfg_h);
        break;
    default:
        throw std::runtime_error("Unknown seas type");
    }

    auto seasop = detail::operator_specifics<SeasQDDiscreteGreenOperator>::make(mesh, cfg_h, *ctx);

    int rank;
    MPI_Comm_rank(seasop->comm(), &rank);

#ifdef PETSC_HAVE_HTOOL
    auto vr = seasop->validate_all();
    double rel_err = vr.err_H_vs_G;
    if (rank == 0) {
        if (rel_err < 0.0) {
            std::cout << "RESULT: SKIP (matrices not available)" << std::endl;
        } else {
            auto fmt_bytes = [](double b) -> std::string {
                char buf[64];
                if (b >= 1e9)
                    std::snprintf(buf, sizeof(buf), "%.2f GB", b / 1e9);
                else
                    std::snprintf(buf, sizeof(buf), "%.2f MB", b / 1e6);
                return buf;
            };

            std::cout << "\n=== H-matrix Validation Report ===" << std::endl;
            std::cout << "Matrix:    " << vr.global_rows << " x " << vr.global_cols
                      << "  (rows x cols)" << std::endl;
            std::cout << "MPI ranks: " << vr.n_ranks << std::endl;

            std::cout << "\nMemory:" << std::endl;
            std::cout << "  G_ (dense): " << fmt_bytes(vr.mem_G_bytes) << std::endl;
            std::cout << "  H_ (htool): " << fmt_bytes(vr.mem_H_bytes) << std::endl;
            if (vr.mem_H_bytes > 0.0) {
                std::cout << "  Compression ratio: "
                          << vr.mem_G_bytes / vr.mem_H_bytes << "x" << std::endl;
            }

            std::cout << "\nMatVec timing (avg over " << vr.n_matvec_reps << " reps):" << std::endl;
            std::cout << "  Gv  (dense): " << vr.time_G_matvec << " s" << std::endl;
            std::cout << "  Hv  (htool): " << vr.time_H_matvec << " s" << std::endl;
            if (vr.time_H_matvec > 0.0) {
                std::cout << "  Speedup H/G: "
                          << vr.time_G_matvec / vr.time_H_matvec << "x" << std::endl;
            }
            std::cout << "\nFull solver: " << vr.time_solver << " s" << std::endl;

            std::cout << "\nRelative errors (||.|| / ||Gv||):" << std::endl;
            std::cout << "  ||Hv - Gv||:          " << vr.err_H_vs_G      << std::endl;
            std::cout << "  ||Gv - solver(v)||:   " << vr.err_G_vs_solver << std::endl;
            std::cout << "  ||Hv - solver(v)||:   " << vr.err_H_vs_solver << std::endl;

            std::cout << std::endl;
            if (rel_err < cfg_h.hmatrix_config.rtol) {
                std::cout << "RESULT: PASS  (||Hv-Gv||/||Gv||=" << rel_err
                          << " < rtol=" << cfg_h.hmatrix_config.rtol << ")" << std::endl;
            } else {
                std::cout << "RESULT: FAIL  (||Hv-Gv||/||Gv||=" << rel_err
                          << " >= rtol=" << cfg_h.hmatrix_config.rtol << ")" << std::endl;
            }
        }
    }
    return rel_err;
#else
    if (rank == 0) {
        std::cout << "RESULT: SKIP (PETSc built without HTools)" << std::endl;
    }
    return -1.0;
#endif
}

} // namespace tndm
