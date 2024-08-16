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
#include "form/BC.h"
#include "io/GMSHParser.h"
#include "io/GlobalSimplexMeshBuilder.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "parallel/Affinity.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/Simplex.h"
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

void generateGlobalMesh(std::unique_ptr<GlobalSimplexMesh<DomainDimension>>& globalMesh,const Config& cfg, int rank, int procs) {
    
    if (cfg.mesh_file) {
        bool ok = false;
        
       

        GlobalSimplexMeshBuilder<DomainDimension> builder;
        
        if (rank == 0) {
            GMSHParser parser(&builder);
            ok = parser.parseFile(*cfg.mesh_file);
            if (!ok) {
                std::cerr << *cfg.mesh_file << std::endl << parser.getErrorMessage();
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

    } else if (cfg.generate_mesh && cfg.resolution) {
        auto meshGen = cfg.generate_mesh->create(*cfg.resolution, PETSC_COMM_WORLD);

        globalMesh = meshGen.uniformMesh();
    }

    if (!globalMesh) {
        PetscFinalize();
        throw std::runtime_error("You must either provide a valid mesh file or provide the mesh generation config (including the resolution parameter).");
        

    } 
    //std::cout << "im here 1" << std::endl;
}
/*
void printSimplexBaseVectorWithRank(std::vector<LocalFaces<1>> vector, int rank) {
    std::cout << "MPI Rank " << rank << ": SimplexBase Vector:" << std::endl;
    for (std::size_t i = 0; i < vector.size(); ++i) {
        std::cout << i << ": ";
        for (const auto& _M_elems : vector[i]) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
}

*/

std::string BCToString(BC bc) {
    switch (bc) {
        case BC::None:
            return "None";
        case BC::Fault:
            return "fault";
        case BC::Dirichlet:
        
            return "Dirichlet";
        default:
            return "Unknown";
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

    if (rank == 0) {
        Banner::standard(std::cout, affinity);
    }

    std::unique_ptr<GlobalSimplexMesh<DomainDimension>> globalMesh;
    generateGlobalMesh( globalMesh,*cfg,  rank, procs);
    //std::cout <<rank << "im here 2" << std::endl;
    globalMesh->repartition();
    
    auto mesh = globalMesh->getLocalMesh(1);
    //std::cout <<rank <<"im here 3" << std::endl;
    //auto const& facets = mesh.facets();
    //auto boundaryData = dynamic_cast<BoundaryData const*>(facets.data());
    //auto aa=boundaryData->getBoundaryConditions()[0];
    //std::cout <<rank << "im here 4" << std::endl;
    auto aa=mesh.get();
    const auto& facets = aa->facets();
    const auto& face=facets.faces();
    const auto& vertices =aa->vertices();
    auto boundaryData = dynamic_cast<BoundaryData const*>(facets.data());
    auto aaa=boundaryData->getBoundaryConditions();
    //printSimplexBaseVectorWithRank(facets,rank);
    std::cout << "MPI Rank " << rank << ": BC Vector: ";
    for (std::size_t i = 0; i < aaa.size(); ++i) {
        std::cout <<"rank:"<<rank<<"-"<< i << ": " << BCToString(aaa[i]) << std::endl;
    }

    auto bcMapping=globalMesh->returnBCMapping();


    
   
    for (const auto& row : bcMapping) {
        std::cout << "bcMapping for rank " << rank << " is: ";
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
       
   


/*
    for (const auto& simplex : vertices) {
        simplex.print(rank);
    }

    */
    std::cout << std::endl;
    solveSEASProblem(*mesh, *cfg);

    PetscErrorCode ierr = PetscFinalize();

    return ierr;
}
