#include "common/CmdLine.h"
#include "common/MeshConfig.h"
#include "config.h"
#include "pc/register.h"
#include "tandem/SEAS.h"
#include "tandem/SeasConfig.h"

#include "io/GMSHParser.h"
#include "io/GlobalSimplexMeshBuilder.h"
#include "mesh/GlobalSimplexMesh.h"
#include "parallel/Affinity.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"

#include <argparse.hpp>
#include <mpi.h>
#include <petscsys.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

using namespace tndm;

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

    argparse::ArgumentParser program("gf-hmatrix-visualize");
    program.add_argument("--petsc").help("PETSc options, must be passed last!");
    program.add_argument("config").help("Configuration file (.toml) — same as tandem, "
                                        "mode must be QDGreen, [hmatrix] section optional.");
    program.add_argument("--output")
        .help("Output file prefix for the leaf-structure CSV (default: hmatrix_structure)")
        .default_value(std::string("hmatrix_structure"));

    auto makePathRelativeToConfig =
        MakePathRelativeToOtherPath([&program]() { return program.get("config"); });

    TableSchema<Config> schema;
    setConfigSchema(schema, makePathRelativeToConfig);

    std::optional<Config> cfg = readFromConfigurationFileAndCmdLine(schema, program, argc, argv);
    if (!cfg) {
        return -1;
    }

    if (cfg->mode != SeasMode::QuasiDynamicDiscreteGreen) {
        std::cerr << "gf-hmatrix-visualize: mode must be QDGreen" << std::endl;
        return -1;
    }

    CHKERRQ(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));
    CHKERRQ(register_PCs());
    CHKERRQ(register_KSPs());

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

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
        if (!ok) {
            PetscFinalize();
            return -1;
        }
        globalMesh = builder.create(PETSC_COMM_WORLD);
        int procs;
        MPI_Comm_size(PETSC_COMM_WORLD, &procs);
        if (procs > 1) {
            globalMesh->repartitionByHash();
        }
    } else {
        std::cerr << "A mesh_file is required." << std::endl;
        PetscFinalize();
        return -1;
    }
    globalMesh->repartition();
    auto mesh = globalMesh->getLocalMesh(1);

    const std::string output_prefix = program.get<std::string>("--output");
    dumpGFHMatrixStructure(*mesh, *cfg, output_prefix);

    return static_cast<int>(PetscFinalize());
}
