#include "localoperator/Poisson.h"
#include "common/CmdLine.h"
#include "common/PetscSolver.h"
#include "common/PetscUtil.h"
#include "common/PoissonScenario.h"
#include "common/Scenario.h"
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
#include "util/Stopwatch.h"

#include <argparse.hpp>
#include <mpi.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <stdexcept>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace tndm;

struct Config {
    std::optional<double> resolution;
    std::optional<std::string> output;
    std::optional<std::string> mesh_file;
    PoissonScenarioConfig problem;
    std::optional<GenMeshConfig<DomainDimension>> generate_mesh;
};

void static_problem(LocalSimplexMesh<DomainDimension> const& mesh, Config const& cfg) {
    tndm::Stopwatch sw;

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    auto scenario = std::make_unique<PoissonScenario>(cfg.problem);
    Curvilinear<DomainDimension> cl(mesh, scenario->transform(), PolynomialDegree);

    auto poisson = std::make_unique<tmp::Poisson>(cl, scenario->coefficient());
    if (scenario->force()) {
        poisson->set_force(*scenario->force());
    }
    if (scenario->boundary()) {
        poisson->set_dirichlet(*scenario->boundary());
    }
    if (scenario->slip() && cfg.problem.ref_normal) {
        poisson->set_slip(*scenario->slip(), *cfg.problem.ref_normal);
    }

    auto dgop =
        DGOperator<DomainDimension, tmp::Poisson>(mesh, std::move(poisson), PETSC_COMM_WORLD);

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

    auto numeric = solver.solution(dgop.lop().space());
    auto solution = scenario->solution();
    if (solution) {
        double error = tndm::Error<DomainDimension>::L2(cl, numeric, *scenario->solution(), 0,
                                                        PETSC_COMM_WORLD);
        if (rank == 0) {
            std::cout << "L2 error (new): " << error << std::endl;
        }
    }

    if (cfg.output) {
        VTUWriter<DomainDimension> writer(PolynomialDegree, true, PETSC_COMM_WORLD);
        auto adapter = CurvilinearVTUAdapter(cl, dgop.numLocalElements());
        auto piece = writer.addPiece(adapter);
        piece.addPointData("u", numeric);
        writer.write(*cfg.output);
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

    argparse::ArgumentParser program("poisson");
    program.add_argument("--petsc").help("PETSc options, must be passed last!");
    program.add_argument("config").help("Configuration file (.toml)");

    auto makePathRelativeToConfig = [&program](std::string_view path) {
        auto newPath = fs::path(program.get("config")).parent_path();
        newPath /= fs::path(path);
        return newPath;
    };

    TableSchema<Config> schema;
    schema.add_value("resolution", &Config::resolution)
        .validator([](auto&& x) { return x > 0; })
        .help("Non-negative resolution parameter");
    schema.add_value("output", &Config::output).help("Output file name");
    schema.add_value("mesh_file", &Config::mesh_file)
        .converter(makePathRelativeToConfig)
        .validator([](std::string const& path) { return fs::exists(fs::path(path)); });
    auto& problemSchema = schema.add_table("problem", &Config::problem);
    problemSchema.add_value("lib", &PoissonScenarioConfig::lib)
        .converter(makePathRelativeToConfig)
        .validator([](std::string const& path) { return fs::exists(fs::path(path)); });
    problemSchema.add_value("warp", &PoissonScenarioConfig::warp);
    problemSchema.add_value("force", &PoissonScenarioConfig::force);
    problemSchema.add_value("boundary", &PoissonScenarioConfig::boundary);
    problemSchema.add_value("slip", &PoissonScenarioConfig::slip);
    problemSchema.add_value("coefficient", &PoissonScenarioConfig::coefficient);
    problemSchema.add_value("solution", &PoissonScenarioConfig::solution);
    problemSchema.add_array("ref_normal", &PoissonScenarioConfig::ref_normal).of_values();
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

    static_problem(*mesh, *cfg);

    ierr = PetscFinalize();

    return ierr;
}
