#include "common/Banner.h"
#include "common/CmdLine.h"
#include "common/MeshConfig.h"
#include "config.h"
#include "pc/register.h"
#include "tandem/AdaptiveOutputStrategy.h"
#include "tandem/FrictionConfig.h"
#include "tandem/SEAS.h"
#include "tandem/SeasConfig.h"
#include "tandem/SeasScenario.h"

#include "io/GlobalSimplexMeshBuilder.h"
#include "io/MeshParser.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "parallel/Affinity.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"

#include <argparse.hpp>
#include <mpi.h>
#include <petscsys.h>
#include <petscsystypes.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace tndm;

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

    argparse::ArgumentParser program("tandem");
    program.add_argument("--petsc").help("PETSc options, must be passed last!");
    program.add_argument("config").help("Configuration file (.toml)");

    auto makePathRelativeToConfig =
        MakePathRelativeToOtherPath([&program]() { return program.get("config"); });

    TableSchema<Config> schema;
    setConfigSchema(schema, makePathRelativeToConfig);

    std::optional<Config> cfg = readFromConfigurationFileAndCmdLine(schema, program, argc, argv);
    if (!cfg) {
        return -1;
    }

    if (cfg->mode == SeasMode::QuasiDynamicDiscreteGreen && !cfg->boundary_linear) {
        std::cerr << "Discrete Green's function can only be used for linear Dirichlet boundaries."
                  << std::endl;
        return -1;
    }

    CHKERRQ(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));
    CHKERRQ(register_PCs());
    CHKERRQ(register_KSPs());

    int rank, procs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &procs);

    auto node_mask = affinity.to_string(affinity.worker_mask_on_node(PETSC_COMM_WORLD));
    if (rank == 0) {
        Banner::standard(std::cout, affinity, node_mask);
    }

    std::unique_ptr<GlobalSimplexMesh<DomainDimension>> globalMesh;
    if (cfg->mesh_file) {
        bool ok = false;
        GlobalSimplexMeshBuilder<DomainDimension> builder;
        std::string meshError;
        if (rank == 0) {
            if (MeshParser::isH5Format(*cfg->mesh_file)) {
#ifdef ENABLE_HDF5
                if constexpr (DomainDimension != 3) {
                    meshError = "H5 mesh format is only supported for 3D problems.";
                }
#else
                meshError = "HDF5 mesh support is not enabled.";
#endif
            }
            if (meshError.empty()) {
                auto parser = MeshParser::create(*cfg->mesh_file, &builder);
                if (!parser) {
                    meshError = "Unsupported mesh file format: " + *cfg->mesh_file;
                } else {
                    ok = parser->parseFile(*cfg->mesh_file);
                    if (!ok) {
                        meshError = *cfg->mesh_file + "\n" + std::string(parser->getErrorMessage());
                    }
                }
            } else {
                ok = false;
            }
        }
        MPI_Bcast(&ok, 1, MPI_CXX_BOOL, 0, PETSC_COMM_WORLD);
        if (!ok) {
            if (rank == 0) {
                std::cerr << meshError << std::endl;
            }
            PetscFinalize();
            return -1;
        }
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

    solveSEASProblem(*mesh, *cfg);

    PetscErrorCode ierr = PetscFinalize();

    return ierr;
}
