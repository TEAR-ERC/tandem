#include "form/BC.h"
#include "io/GMSHParser.h"
#include "io/GlobalSimplexMeshBuilder.h"
#include "io/VTUAdapter.h"
#include "io/VTUWriter.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/LocalSimplexMesh.h"

#include <argparse.hpp>
#include <exception>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <string>
#include <utility>
#include <vector>

using namespace tndm;

template <std::size_t D>
auto load_mesh(std::string const& mesh_file) -> std::unique_ptr<GlobalSimplexMesh<D>> {
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

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
        if (rank == 0) {
            std::cerr << "You must either provide a valid mesh file." << std::endl;
        }
        return nullptr;
    }
    auto globalMesh = builder.create(MPI_COMM_WORLD);
    if (procs > 1) {
        // ensure initial element distribution for metis
        globalMesh->repartitionByHash();
    }
    globalMesh->repartition();
    return globalMesh;
}

template <std::size_t D>
auto find_facets(LocalSimplexMesh<D> const& mesh)
    -> std::pair<std::vector<std::size_t>, std::vector<int>> {
    auto const& facets = mesh.facets();
    auto boundaryData = dynamic_cast<ScalarMeshData<int> const*>(facets.pTagData());
    if (!boundaryData) {
        throw std::runtime_error("Boundary conditions not set.");
    }

    std::size_t numLocalFacets = mesh.facets().localSize();
    std::vector<std::size_t> theFctNos;
    std::vector<int> theBcs;
    theFctNos.reserve(numLocalFacets);
    theBcs.reserve(numLocalFacets);

    auto const& bc = boundaryData->getData();
    for (std::size_t fctNo = 0; fctNo < numLocalFacets; ++fctNo) {
        if (bc[fctNo] != static_cast<int>(BC::None)) {
            theFctNos.emplace_back(fctNo);
            theBcs.emplace_back(bc[fctNo]);
        }
    }
    return std::make_pair(std::move(theFctNos), std::move(theBcs));
}

template <std::size_t D>
void write_bcs(LocalSimplexMesh<D> const& mesh, unsigned N, std::vector<std::size_t> fctNos,
               std::vector<int> bcs, std::string const& prefix) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto cl = std::make_shared<Curvilinear<D>>(
        mesh, [](typename Curvilinear<D>::vertex_t const& v) { return v; }, N);
    auto adapter = CurvilinearBoundaryVTUAdapter(mesh, cl, fctNos);
    auto writer = VTUWriter<D - 1u>(N, true, MPI_COMM_WORLD);

    auto& piece = writer.addPiece(adapter);
    piece.addCellData("bc", bcs.data(), bcs.size());
    writer.write(prefix);
}

template <std::size_t D>
bool check_bc(std::string const& mesh_file, std::string const& output, unsigned N) {
    auto globalMesh = load_mesh<D>(mesh_file);
    if (!globalMesh) {
        return false;
    }
    auto mesh = globalMesh->getLocalMesh(0);
    auto [theFctNos, theBcs] = find_facets<D>(*mesh);
    write_bcs<D>(*mesh, N, theFctNos, theBcs, output);
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    argparse::ArgumentParser program("static");
    program.add_argument("-N", "--degree")
        .help("Polynomial degree for geometry approximation")
        .default_value(1ul)
        .action([](std::string const& value) { return std::stoul(value); });
    program.add_argument("dim")
        .help("Simplex dimension (D=2: triangle, D=3: tet)")
        .action([](std::string const& value) { return std::stoul(value); });
    program.add_argument("mesh_file").help(".msh file");
    program.add_argument("output").help("Prefix of VTU output");

    try {
        program.parse_args(argc, argv);
    } catch (std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        MPI_Finalize();
        return 0;
    }

    auto N = program.get<unsigned long>("-N");
    auto D = program.get<unsigned long>("dim");
    auto mesh_file = program.get<std::string>("mesh_file");
    auto output = program.get<std::string>("output");

    if (D == 2u) {
        check_bc<2u>(mesh_file, output, N);
    } else if (D == 3u) {
        check_bc<3u>(mesh_file, output, N);
    } else {
        std::cerr << "Unsupported dimension " << D << std::endl;
    }

    MPI_Finalize();
    return 0;
}
