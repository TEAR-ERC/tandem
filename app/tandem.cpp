#include "common/CmdLine.h"
#include "tandem/Config.h"
#include "tandem/SEAS.h"
#include "tandem/Static.h"

#include "util/Schema.h"

#include <argparse.hpp>
#include <mpi.h>
#include <petscsys.h>

#include <filesystem>

namespace fs = std::filesystem;
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

    argparse::ArgumentParser program("tandem");
    program.add_argument("--petsc").help("PETSc options, must be passed last!");
    program.add_argument("config").help("Configuration file (.toml)");

    TableSchema<Config> schema;
    schema.add_value("resolution", &Config::resolution)
        .validator([](auto&& x) { return x > 0; })
        .help("Non-negative resolution parameter");
    schema.add_value("final_time", &Config::final_time)
        .validator([](auto&& x) { return x > 0; })
        .help("Non-negative final time of simulation");
    schema.add_value("output", &Config::output).help("Output file name");
    schema.add_value("output_interval", &Config::output_interval)
        .validator([](auto&& x) { return x > 0; })
        .help("Non-negative output interval");
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
    problemSchema.add_value("slip", &ProblemConfig::slip);
    problemSchema.add_value("lam", &ProblemConfig::lam);
    problemSchema.add_value("mu", &ProblemConfig::mu);
    problemSchema.add_value("solution", &ProblemConfig::solution);
    auto& genMeshSchema = schema.add_table("generate_mesh", &Config::generate_mesh);
    GenMeshConfig<DomainDimension>::setSchema(genMeshSchema);

    std::optional<Config> cfg = readFromConfigurationFileAndCmdLine(schema, program, argc, argv);
    if (!cfg) {
        return -1;
    }

    PetscErrorCode ierr;
    CHKERRQ(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));

    // solveStaticProblem(*cfg);
    solveSEASProblem(*cfg);

    ierr = PetscFinalize();

    return ierr;
}
