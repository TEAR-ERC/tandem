#include "poisson/Poisson.h"
#include "common/CmdLine.h"
#include "common/Scenario.h"
#include "config.h"
#include "poisson/Scenario.h"

#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "io/VTUWriter.h"
#include "mesh/GenMesh.h"
#include "mesh/MeshData.h"
#include "tensor/Tensor.h"
#include "util/Schema.h"
#include "util/Stopwatch.h"

#include <argparse.hpp>
#include <fstream>
#include <mpi.h>
#include <ostream>
#include <petscerror.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <stdexcept>
#include <toml.hpp>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <tuple>

namespace fs = std::filesystem;
using namespace tndm;

struct Config {
    double resolution;
    std::optional<std::string> output;
    ProblemConfig problem;
    GenMeshConfig<DomainDimension> generate_mesh;
};

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

    TableSchema<Config> schema;
    schema.add_value("resolution", &Config::resolution)
        .validator([](auto&& x) { return x > 0; })
        .help("Non-negative resolution parameter");
    schema.add_value("output", &Config::output).help("Output file name");
    auto& problemSchema = schema.add_table("problem", &Config::problem);
    problemSchema.add_value("lib", &ProblemConfig::lib)
        .converter([&program](std::string_view path) {
            auto newPath = fs::path(program.get("config")).parent_path();
            newPath /= fs::path(path);
            return newPath;
        })
        .validator([](std::string const& path) { return fs::exists(fs::path(path)); });
    problemSchema.add_value("warp", &ProblemConfig::warp);
    problemSchema.add_value("force", &ProblemConfig::force);
    problemSchema.add_value("boundary", &ProblemConfig::boundary);
    problemSchema.add_value("coefficient", &ProblemConfig::coefficient);
    problemSchema.add_value("solution", &ProblemConfig::solution);
    auto& genMeshSchema = schema.add_table("generate_mesh", &Config::generate_mesh);
    GenMeshConfig<DomainDimension>::setSchema(genMeshSchema);

    std::optional<Config> cfg = readFromConfigurationFileAndCmdLine(schema, program, argc, argv);
    if (!cfg) {
        return -1;
    }

    PetscErrorCode ierr;
    CHKERRQ(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));

    auto scenario = std::make_unique<LuaScenario>(cfg->problem);

    int rank, procs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &procs);

    auto meshGen = cfg->generate_mesh.create(cfg->resolution, PETSC_COMM_WORLD);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();
    auto mesh = globalMesh->getLocalMesh(1);

    Curvilinear<DomainDimension> cl(*mesh, scenario->transform(), PolynomialDegree);

    tndm::Stopwatch sw;

    sw.start();
    Poisson poisson(*mesh, cl,
                    std::make_unique<tndm::ModalRefElement<DomainDimension>>(PolynomialDegree),
                    MinQuadOrder(), PETSC_COMM_WORLD, scenario->coefficient());
    std::cout << "Constructed Poisson after " << sw.split() << std::endl;

    Mat A;
    Vec b, x, y;
    KSP ksp;

    {
        auto interface = poisson.interfacePetsc();
        CHKERRQ(interface.createA(&A));
        CHKERRQ(interface.createb(&b));
    }

    CHKERRQ(poisson.assemble(A));
    CHKERRQ(poisson.rhs(b, scenario->force(), scenario->boundary()));
    std::cout << "Assembled after " << sw.split() << std::endl;

    CHKERRQ(VecDuplicate(b, &x));
    CHKERRQ(VecDuplicate(b, &y));
    CHKERRQ(VecSet(x, 1.0));
    CHKERRQ(MatMult(A, x, y));
    PetscReal l2norm;
    CHKERRQ(VecNorm(y, NORM_2, &l2norm));
    std::cout << "A*1 norm: " << l2norm << std::endl;

    CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ksp));
    CHKERRQ(KSPSetType(ksp, KSPCG));
    CHKERRQ(KSPSetOperators(ksp, A, A));
    CHKERRQ(KSPSetTolerances(ksp, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    CHKERRQ(KSPSetFromOptions(ksp));

    /* If you want to use the BAIJ operator on the finest level, forcefully insert it */
    /*
    {
        KSP      smoother;
        PC       pc;
        PetscInt nlevels;

        ierr = KSPSetUp(ksp);CHKERRQ(ierr);
        ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
        ierr = PCMGGetLevels(pc,&nlevels);CHKERRQ(ierr);
        ierr = PCMGGetSmoother(pc,nlevels-1,&smoother);CHKERRQ(ierr);
        ierr = KSPSetOperators(smoother, Aaij, A);CHKERRQ(ierr);
        ierr = KSPSetUp(smoother);CHKERRQ(ierr);
    }
    */

    CHKERRQ(KSPSolve(ksp, b, x));
    std::cout << "Solved after " << sw.split() << std::endl;
    PetscReal rnorm;
    PetscInt its;
    CHKERRQ(KSPGetResidualNorm(ksp, &rnorm));
    CHKERRQ(KSPGetIterationNumber(ksp, &its));
    if (rank == 0) {
        std::cout << "Residual norm: " << rnorm << std::endl;
        std::cout << "Iterations: " << its << std::endl;
    }

    CHKERRQ(KSPDestroy(&ksp));
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(VecDestroy(&b));

    auto numeric = poisson.finiteElementFunction(x);
    double error =
        tndm::Error<DomainDimension>::L2(cl, numeric, *scenario->solution(), 0, PETSC_COMM_WORLD);

    if (rank == 0) {
        std::cout << "L2 error: " << error << std::endl;
    }

    if (cfg->output) {
        VTUWriter<DomainDimension> writer(PolynomialDegree, true, PETSC_COMM_WORLD);
        auto piece = writer.addPiece(cl, poisson.numLocalElements());
        piece.addPointData("u", numeric);
        piece.addPointData("K", poisson.discreteK());
        writer.write(*cfg->output);
    }

    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y));

    ierr = PetscFinalize();

    return ierr;
}
