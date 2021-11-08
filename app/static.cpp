#include "common/Banner.h"
#include "common/CmdLine.h"
#include "common/ElasticityScenario.h"
#include "common/MGConfig.h"
#include "common/MeshConfig.h"
#include "common/PetscLinearSolver.h"
#include "common/PetscUtil.h"
#include "common/PoissonScenario.h"
#include "config.h"
#include "form/DGCurvilinearCommon.h"
#include "mesh/LocalSimplexMesh.h"
#include "pc/register.h"

#include "form/DGOperator.h"
#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "io/GMSHParser.h"
#include "io/GlobalSimplexMeshBuilder.h"
#include "io/VTUAdapter.h"
#include "io/VTUWriter.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "parallel/Affinity.h"
#include "tensor/Managed.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"
#include "util/Stopwatch.h"

#include <argparse.hpp>
#include <limits>
#include <mpi.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <stdexcept>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace tndm;

struct Config {
    std::optional<double> resolution;
    DGMethod method;
    bool matrix_free;
    bool test_matrix_free;
    MGStrategy mg_strategy;
    unsigned mg_coarse_level;
    int profile;
    std::optional<std::string> output;
    std::optional<std::string> mesh_file;
    std::optional<PoissonScenarioConfig> poisson;
    std::optional<ElasticityScenarioConfig> elasticity;
    std::optional<GenMeshConfig<DomainDimension>> generate_mesh;
};

template <class Scenario>
void static_problem(LocalSimplexMesh<DomainDimension> const& mesh, Scenario const& scenario,
                    Config const& cfg) {
    tndm::Stopwatch sw;
    double time;

    PetscLogStage solve;
    PetscLogStageRegister("solve", &solve);

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    auto cl = std::make_shared<Curvilinear<DomainDimension>>(mesh, scenario.transform(),
                                                             PolynomialDegree);

    auto lop = scenario.make_local_operator(cl, cfg.method);
    auto topo = std::make_shared<DGOperatorTopo>(mesh, PETSC_COMM_WORLD);
    auto dgop = DGOperator(topo, std::move(lop));

    const auto reduce_number = [&topo](std::size_t number) {
        std::size_t number_global;
        MPI_Reduce(&number, &number_global, 1, mpi_type_t<std::size_t>(), MPI_SUM, 0, topo->comm());
        return number_global;
    };

    if (cfg.test_matrix_free) {
        auto A = std::make_unique<PetscDGShell>(dgop);
        Vec x, y;
        CHKERRTHROW(MatCreateVecs(A->mat(), &x, &y));
        PetscRandom rctx;
        CHKERRTHROW(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
        CHKERRTHROW(VecSetRandom(x, rctx));
        CHKERRTHROW(PetscRandomDestroy(&rctx));

        std::size_t flops = dgop.flops_apply();
        auto flops_global = reduce_number(flops);

        constexpr int nrepeat = 100;
        Stopwatch sw;
        sw.start();
        for (int i = 0; i < nrepeat; ++i) {
            MatMult(A->mat(), x, y);
        }
        auto time = sw.stop();
        if (rank == 0) {
            time /= nrepeat;
            std::cout << "Shell time: " << time << " s" << std::endl;
            std::cout << "Shell flops: " << flops_global << std::endl;
            std::cout << "Shell GFLOPS: " << flops_global / time * 1e-9 << std::endl;
        }
        CHKERRTHROW(VecDestroy(&x));
        CHKERRTHROW(VecDestroy(&y));
    }

    std::size_t num_dofs_domain = reduce_number(dgop.number_of_local_dofs());

    double local_mesh_size = cl->local_mesh_size();
    double mesh_size;
    MPI_Reduce(&local_mesh_size, &mesh_size, 1, mpi_type_t<double>(), MPI_MAX, 0, topo->comm());

    if (rank == 0) {
        std::cout << "DOFs: " << num_dofs_domain << std::endl;
        std::cout << "Mesh size: " << mesh_size << std::endl;
    }

    sw.start();
    auto solver =
        PetscLinearSolver(dgop, cfg.matrix_free, MGConfig(cfg.mg_coarse_level, cfg.mg_strategy));
    time = sw.stop();
    if (rank == 0) {
        std::cout << "Assembly: " << time << " s" << std::endl;
    }

    sw.start();
    if (cfg.profile > 0) {
        solver.solve();
    } else {
        solver.warmup();
    }
    time = sw.stop();
    if (rank == 0) {
        std::cout << "Solver warmup: " << time << " s" << std::endl;
    }

    PetscLogStagePush(solve);
    if (cfg.profile > 0) {
        double avg_time = 0.0;
        double max_time = 0.0;
        double min_time = std::numeric_limits<double>::max();
        for (int p = 0; p < cfg.profile; ++p) {
            sw.start();
            solver.solve();
            time = sw.stop();
            avg_time += time;
            max_time = std::max(max_time, time);
            min_time = std::min(min_time, time);
        }
        avg_time /= cfg.profile;
        if (rank == 0) {
            std::cout << "Solve (min): " << min_time << " s" << std::endl;
            std::cout << "Solve (avg): " << avg_time << " s" << std::endl;
            std::cout << "Solve (max): " << max_time << " s" << std::endl;
        }
    } else {
        sw.start();
        solver.solve();
        time = sw.stop();
        if (rank == 0) {
            std::cout << "Solve: " << time << " s" << std::endl;
        }
    }
    PetscLogStagePop();
    if (!solver.is_converged()) {
        std::cout << "Solver did not converge." << std::endl;
        return;
    }

    PetscReal rnorm;
    PetscInt its;
    CHKERRTHROW(KSPGetResidualNorm(solver.ksp(), &rnorm));
    CHKERRTHROW(KSPGetIterationNumber(solver.ksp(), &its));
    if (rank == 0) {
        std::cout << "Residual norm: " << rnorm << std::endl;
        std::cout << "Iterations: " << its << std::endl;
    }

    auto numeric = dgop.solution(solver.x());
    auto solution = scenario.solution();
    if (solution) {
        double error =
            tndm::Error<DomainDimension>::L2(*cl, numeric, *solution, 0, PETSC_COMM_WORLD);
        if (rank == 0) {
            std::cout << "L2 error: " << error << std::endl;
        }
    }
    auto solution_jacobian = scenario.solution_jacobian();
    if (solution_jacobian) {
        double error = tndm::Error<DomainDimension>::H1_semi(*cl, numeric, *solution_jacobian, 0,
                                                             PETSC_COMM_WORLD);
        if (rank == 0) {
            std::cout << "H1-semi error: " << error << std::endl;
        }
    }

    if (cfg.output) {
        auto coeffs = dgop.params();
        VTUWriter<DomainDimension> writer(PolynomialDegree, true, PETSC_COMM_WORLD);
        auto adapter = CurvilinearVTUAdapter(cl, dgop.num_local_elements());
        auto piece = writer.addPiece(adapter);
        piece.addPointData(numeric);
        piece.addPointData(coeffs);
        writer.write(*cfg.output);
    }
}

int main(int argc, char** argv) {
    auto affinity = Affinity();

    int pArgc = 0;
    char** pArgv = nullptr;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--petsc") == 0) {
            pArgc = argc - i;
            pArgv = argv + i;
            argc = i;
            break;
        }
    }

    argparse::ArgumentParser program("static");
    program.add_argument("--petsc").help("PETSc options, must be passed last!");
    program.add_argument("config").help("Configuration file (.toml)");

    auto makePathRelativeToConfig =
        MakePathRelativeToOtherPath([&program]() { return program.get("config"); });

    TableSchema<Config> schema;
    schema.add_value("resolution", &Config::resolution)
        .validator([](auto&& x) { return x > 0; })
        .help("Non-negative resolution parameter");
    schema.add_value("method", &Config::method)
        .converter([](std::string_view value) {
            if (iEquals(value, "ip")) {
                return DGMethod::IP;
            } else if (iEquals(value, "br2")) {
                return DGMethod::BR2;
            } else {
                return DGMethod::Unknown;
            }
        })
        .default_value(DGMethod::IP)
        .validator([](DGMethod const& type) { return type != DGMethod::Unknown; });
    schema.add_value("matrix_free", &Config::matrix_free).default_value(false);
    schema.add_value("test_matrix_free", &Config::test_matrix_free).default_value(false);
    schema.add_value("mg_coarse_level", &Config::mg_coarse_level).default_value(1);
    schema.add_value("mg_strategy", &Config::mg_strategy)
        .converter([](std::string_view value) {
            if (iEquals(value, "TwoLevel")) {
                return MGStrategy::TwoLevel;
            } else if (iEquals(value, "Logarithmic")) {
                return MGStrategy::Logarithmic;
            } else if (iEquals(value, "Full")) {
                return MGStrategy::Full;
            } else {
                return MGStrategy::Unknown;
            }
        })
        .default_value(MGStrategy::TwoLevel)
        .validator([](MGStrategy const& type) { return type != MGStrategy::Unknown; });
    schema.add_value("profile", &Config::profile)
        .default_value(0)
        .validator([](auto&& x) { return x >= 0; })
        .help("Run static in profile mode. The parameter controls the amount of repetitions.");
    schema.add_value("output", &Config::output).help("Output file name");
    schema.add_value("mesh_file", &Config::mesh_file)
        .converter(makePathRelativeToConfig)
        .validator(PathExists());
    auto& poissonSchema = schema.add_table("poisson", &Config::poisson);
    PoissonScenarioConfig::setSchema(poissonSchema, makePathRelativeToConfig);
    auto& elasticitySchema = schema.add_table("elasticity", &Config::elasticity);
    ElasticityScenarioConfig::setSchema(elasticitySchema, makePathRelativeToConfig);
    auto& genMeshSchema = schema.add_table("generate_mesh", &Config::generate_mesh);
    GenMeshConfig<DomainDimension>::setSchema(genMeshSchema);

    std::optional<Config> cfg = readFromConfigurationFileAndCmdLine(schema, program, argc, argv);
    if (!cfg) {
        return -1;
    }

    CHKERRQ(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));
    CHKERRQ(register_PCs());
    CHKERRQ(register_KSPs());

    int rank, procs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &procs);

    if (rank == 0) {
        Banner::print_logo_version_and_affinity(std::cout, affinity);
    }

    std::unique_ptr<GlobalSimplexMesh<DomainDimension>> globalMesh;
    if (cfg->mesh_file) {
        bool ok = false;
        GlobalSimplexMeshBuilder<DomainDimension> builder;
        if (rank == 0) {
            GMSHParser parser(&builder);
            ok = parser.parseFile(*cfg->mesh_file);
            if (!ok) {
                std::cerr << *cfg->mesh_file << std::endl << parser.getErrorMessage();
            }
        }
        MPI_Bcast(&ok, 1, MPI_CXX_BOOL, 0, PETSC_COMM_WORLD);
        if (ok) {
            globalMesh = builder.create(PETSC_COMM_WORLD);
        }
        if (procs > 1) {
            // ensure initial element distribution for metis
            globalMesh->repartitionByHash();
        }
    } else if (cfg->generate_mesh && cfg->resolution) {
        auto meshGen = cfg->generate_mesh->create(*cfg->resolution, PETSC_COMM_WORLD);
        globalMesh = meshGen.uniformMesh();
    }
    if (!globalMesh) {
        std::cerr
            << "You must either provide a valid mesh file or provide the mesh generation config "
               "(including the resolution parameter)."
            << std::endl;
        PetscFinalize();
        return -1;
    }
    globalMesh->repartition();
    auto mesh = globalMesh->getLocalMesh(1);

    if (cfg->poisson && !cfg->elasticity) {
        auto scenario = PoissonScenario(*cfg->poisson);
        static_problem(*mesh, scenario, *cfg);
    } else if (!cfg->poisson && cfg->elasticity) {
        auto scenario = ElasticityScenario(*cfg->elasticity);
        static_problem(*mesh, scenario, *cfg);
    } else {
        std::cerr << "Please specify either [poisson] or [elasticity] (but not both)." << std::endl;
    }

    PetscErrorCode ierr = PetscFinalize();

    return ierr;
}
