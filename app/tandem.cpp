#include "geometry/Curvilinear.h"
#include "io/VTUWriter.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/MeshData.h"
#include "util/Range.h"

#include <argparse.hpp>
#include <mpi.h>
#include <parmetis.h>

#include <cmath>
#include <iostream>

using tndm::BoundaryData;
using tndm::Curvilinear;
using tndm::GenMesh;
using tndm::LocalSimplexMesh;
using tndm::Range;
using tndm::VertexData;
using tndm::VTUWriter;

template <std::size_t D, typename Fun>
void writeMesh(std::string const& baseName, std::array<uint64_t, D> const& N, Fun transform,
               int ghostLevels) {
    GenMesh<D> meshGen(N);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();

    auto mesh = globalMesh->getLocalMesh(ghostLevels);

    unsigned degree = 15;
    Curvilinear<D> cl(*mesh, *transform, degree);

    auto boundaryData = dynamic_cast<BoundaryData const*>(mesh->facets().data());
    if (!boundaryData) {
        return;
    }
    std::vector<double> bc(mesh->numElements(), 0.0);
    for (std::size_t fid = 0; fid < mesh->numFacets(); ++fid) {
        auto& eids = mesh->template upward<D - 1>(fid);
        for (auto& eid : eids) {
            bc[eid] += boundaryData->getBoundaryConditions()[fid];
        }
    }

    std::vector<std::size_t> shared(mesh->elements().localSize(), 0);
    for (std::size_t i = 0; i < mesh->elements().localSize(); ++i) {
        shared[i] += mesh->elements().getSharedRanks(i).size();
    }

    VTUWriter<D> writer(degree);
    auto piece = writer.addPiece(cl, mesh->elements().localSize());
    piece.addCellData("BC", bc);
    piece.addCellData("shared", shared);
    writer.write(baseName);
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
    program.add_argument("-o").default_value(std::string("testmesh")).help("Output file name");
    program.add_argument("n")
        .help("Number of elements per dimension")
        .action([](std::string const& value) { return std::stoul(value); });

    try {
        program.parse_args(argc, argv);
    } catch (std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 0;
    }

    const auto D = program.get<int>("-D");
    const auto ghost = program.get<int>("-g");
    const auto out = program.get<std::string>("-o");
    const auto n = program.get<unsigned long>("n");

    if (D == 2) {
        std::array<uint64_t, 2> N = {n, n};
        auto transform = [](std::array<double, 2> const& v) -> std::array<double, 2> {
            double r = 0.5 * (v[0] + 1.0);
            double phi = 0.5 * M_PI * v[1];
            return {r * cos(phi), r * sin(phi)};
        };
        writeMesh(out, N, transform, ghost);
    } else if (D == 3) {
        std::array<uint64_t, 3> N = {n, n, n};
        auto transform = [](std::array<double, 3> const& v) -> std::array<double, 3> {
            double r = 0.5 * (v[0] + 1.0);
            double phi = 0.5 * M_PI * v[1];
            double theta = M_PI * (0.5 * v[2] + 0.25);
            return {r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)};
        };
        writeMesh(out, N, transform, ghost);
    } else {
        std::cerr << "Unsupported simplex dimension: " << D << std::endl;
    }

    MPI_Finalize();

    return 0;
}
