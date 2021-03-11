#include "common/CmdLine.h"
#include "common/ElasticityScenario.h"
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
#include "tensor/Managed.h"
#include "util/Affinity.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"
#include "util/Stopwatch.h"

#include <argparse.hpp>
#include <mpi.h>
#include <petscksp.h>
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
    std::optional<std::string> output;
    std::optional<std::string> mesh_file;
    std::optional<PoissonScenarioConfig> poisson;
    std::optional<ElasticityScenarioConfig> elasticity;
    std::optional<GenMeshConfig<DomainDimension>> generate_mesh;
};

template <class Scenario>
void static_problem(LocalSimplexMesh<DomainDimension> const& mesh, Scenario const& scenario,
                    DGMethod method, bool matrix_free, std::optional<std::string> const& output) {
    tndm::Stopwatch sw;
    double time;

    PetscLogStage solve;
    PetscLogStageRegister("solve", &solve);

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    auto cl = std::make_shared<Curvilinear<DomainDimension>>(mesh, scenario.transform(),
                                                             PolynomialDegree);

    auto lop = scenario.make_local_operator(cl, method);
    auto topo = std::make_shared<DGOperatorTopo>(mesh, PETSC_COMM_WORLD);
    auto dgop = DGOperator(topo, std::move(lop));

    sw.start();
    auto solver = PetscLinearSolver(dgop, matrix_free);
    time = sw.stop();
    if (rank == 0) {
        std::cout << "Assembly: " << time << " s" << std::endl;
    }

    sw.start();
    solver.warmup();
    time = sw.stop();
    if (rank == 0) {
        std::cout << "Solver warmup: " << time << " s" << std::endl;
    }

    PetscLogStagePush(solve);
    sw.start();
    solver.solve();
    time = sw.stop();
    if (rank == 0) {
        std::cout << "Solve: " << time << " s" << std::endl;
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

    if (output) {
        auto coeffs = dgop.coefficients();
        VTUWriter<DomainDimension> writer(PolynomialDegree, true, PETSC_COMM_WORLD);
        auto adapter = CurvilinearVTUAdapter(cl, dgop.numLocalElements());
        auto piece = writer.addPiece(adapter);
        piece.addPointData("u", numeric);
        piece.addPointData("material", coeffs);
        writer.write(*output);
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

    std::cout << "Worker affinity: " << affinity.to_string(affinity.worker_mask()) << std::endl;

    CHKERRQ(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));
    CHKERRQ(register_PCs());

    int rank, procs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &procs);

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
        static_problem(*mesh, scenario, cfg->method, cfg->matrix_free, cfg->output);
    } else if (!cfg->poisson && cfg->elasticity) {
        auto scenario = ElasticityScenario(*cfg->elasticity);
        static_problem(*mesh, scenario, cfg->method, cfg->matrix_free, cfg->output);
    } else {
        std::cerr << "Please specify either [poisson] or [elasticity] (but not both)." << std::endl;
    }

    PetscErrorCode ierr = PetscFinalize();

    return ierr;
}
