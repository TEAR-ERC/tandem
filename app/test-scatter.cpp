#include "io/GMSHParser.h"
#include "io/GlobalSimplexMeshBuilder.h"
#include "mesh/LocalSimplexMesh.h"
#include "parallel/SimpleScatter.h"
#include "util/Stopwatch.h"

#include <argparse.hpp>
#include <mpi.h>

#include <cstddef>
#include <iostream>

using namespace tndm;

template <std::size_t D> void test_scatter(LocalSimplexMesh<D> const& mesh) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const auto& elements = mesh.elements();

    auto test_data = std::vector<std::size_t>(elements.size());
    for (std::size_t elNo = 0; elNo < elements.localSize(); ++elNo) {
        test_data[elNo] = elements.l2cg(elNo);
    }
    auto scatter =
        SimpleScatter<std::size_t>(std::make_shared<ScatterPlan>(elements, MPI_COMM_WORLD));
    scatter.scatter(test_data.data());

    const auto check_scatter = [](const auto& elements, std::vector<std::size_t> const& data) {
        bool ok = true;
        for (std::size_t elNo = 0; elNo < elements.size(); ++elNo) {
            ok = ok && data[elNo] == elements.l2cg(elNo);
        }
        return ok;
    };

    if (!check_scatter(elements, test_data)) {
        std::cerr << "Scatter is incorrect." << std::endl;
    } else {
        const auto nrepeat = 100;
        Stopwatch sw;
        sw.start();
        for (int i = 0; i < nrepeat; ++i) {
            scatter.scatter(test_data.data());
        }
        auto time = sw.stop();
        if (rank == 0) {
            std::cout << "Time per scatter: " << time / nrepeat << std::endl;
        }
    }
}

template <std::size_t D> void test(std::string mesh_file, unsigned long overlap) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    bool ok = false;
    GlobalSimplexMeshBuilder<D> builder;
    if (rank == 0) {
        GMSHParser parser(&builder);
        ok = parser.parseFile(mesh_file);
        if (!ok) {
            std::cerr << mesh_file << std::endl << parser.getErrorMessage();
        }
    }
    MPI_Bcast(&ok, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    if (!ok) {
        return;
    }
    auto globalMesh = builder.create(MPI_COMM_WORLD);
    globalMesh->repartitionByHash();
    globalMesh->repartition();
    auto mesh = globalMesh->getLocalMesh(overlap);

    test_scatter<D>(*mesh);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    argparse::ArgumentParser program("tandem");
    program.add_argument("-D", "--dim")
        .help("Simplex dimension (D=2: triangle, D=3: tet)")
        .default_value(2)
        .action([](std::string const& value) { return std::stoi(value); });
    program.add_argument("-g")
        .help("Ghost levels")
        .default_value(1)
        .action([](std::string const& value) { return std::stoi(value); });
    program.add_argument("mesh_file").help("Mesh file");

    try {
        program.parse_args(argc, argv);
    } catch (std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 0;
    }

    const auto D = program.get<int>("-D");
    const auto ghost = program.get<int>("-g");
    const auto mesh_file = program.get<std::string>("mesh_file");

    if (D == 2) {
        test<2>(mesh_file, ghost);
    } else if (D == 3) {
        test<3>(mesh_file, ghost);
    } else {
        std::cerr << "Unsupported simplex dimension: " << D << std::endl;
    }

    MPI_Finalize();

    return 0;
}
