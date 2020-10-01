#include "common/CmdLine.h"
#include "common/ElasticityScenario.h"
#include "common/MeshConfig.h"
#include "common/PetscSolver.h"
#include "common/PetscUtil.h"
#include "common/PoissonScenario.h"
#include "config.h"
#include "mesh/LocalSimplexMesh.h"

#include "form/BC.h"
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
#include "util/Range.h"
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
    std::optional<std::string> output;
    std::optional<std::string> mesh_file;
    std::optional<PoissonScenarioConfig> poisson;
    std::optional<ElasticityScenarioConfig> elasticity;
    std::optional<GenMeshConfig<DomainDimension>> generate_mesh;
};

template <class Scenario>
void static_problem(LocalSimplexMesh<DomainDimension> const& mesh, Scenario const& scenario,
                    std::optional<std::string> const& output) {
    tndm::Stopwatch sw;

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    Curvilinear<DomainDimension> cl(mesh, scenario.transform(), PolynomialDegree);

    auto lop = scenario.make_local_operator(cl);
    auto dgop = DGOperator<DomainDimension, typename decltype(lop)::element_type>(
        mesh, std::move(lop), PETSC_COMM_WORLD);

    sw.start();
    PetscSolver solver;
    dgop.assemble(solver);
    dgop.rhs(solver);
    std::cout << "Assembled after " << sw.split() << std::endl;

    solver.solve();
    std::cout << "Solved after " << sw.split() << std::endl;

    PetscReal rnorm;
    PetscInt its;
    CHKERRTHROW(KSPGetResidualNorm(solver.ksp(), &rnorm));
    CHKERRTHROW(KSPGetIterationNumber(solver.ksp(), &its));
    if (rank == 0) {
        std::cout << "Residual norm: " << rnorm << std::endl;
        std::cout << "Iterations: " << its << std::endl;
    }

    auto numeric = dgop.solution(solver);
    auto solution = scenario.solution();
    if (solution) {
        double error =
            tndm::Error<DomainDimension>::L2(cl, numeric, *solution, 0, PETSC_COMM_WORLD);
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

    PetscErrorCode ierr;
    CHKERRQ(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));

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
        static_problem(*mesh, scenario, cfg->output);
    } else if (!cfg->poisson && cfg->elasticity) {
        auto scenario = ElasticityScenario(*cfg->elasticity);
        static_problem(*mesh, scenario, cfg->output);
    } else {
        std::cerr << "Please specify either [poisson] or [elasticity] (but not both)." << std::endl;
    }

    ierr = PetscFinalize();

    return ierr;
}
