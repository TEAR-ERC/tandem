#include "geometry/Curvilinear.h"
#include "io/VTUAdapter.h"
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

using tndm::BC;
using tndm::BoundaryData;
using tndm::Curvilinear;
using tndm::CurvilinearVTUAdapter;
using tndm::GenMesh;
using tndm::LocalSimplexMesh;
using tndm::Range;
using tndm::VertexData;
using tndm::VTUWriter;

template <std::size_t D, typename Fun>
void writeMesh(std::string const& baseName, GenMesh<D> const& meshGen, Fun transform,
               int ghostLevels) {
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();

    auto mesh = globalMesh->getLocalMesh(ghostLevels);

    unsigned degree = 1;
    Curvilinear<D> cl(*mesh, *transform, degree);

    auto boundaryData = dynamic_cast<BoundaryData const*>(mesh->facets().data());
    if (!boundaryData) {
        return;
    }
    std::vector<double> bc(mesh->numElements(), 0.0);
    for (std::size_t fid = 0; fid < mesh->numFacets(); ++fid) {
        auto& eids = mesh->template upward<D - 1>(fid);
        for (auto& eid : eids) {
            bc[eid] += static_cast<int>(boundaryData->getBoundaryConditions()[fid]);
        }
    }

    std::vector<std::size_t> shared(mesh->elements().localSize(), 0);
    for (std::size_t i = 0; i < mesh->elements().localSize(); ++i) {
        shared[i] += mesh->elements().getSharedRanks(i).size();
    }

    VTUWriter<D> writer(degree);
    auto adapter = CurvilinearVTUAdapter(cl, mesh->elements().localSize());
    auto piece = writer.addPiece(adapter);
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
        auto transform = [](std::array<double, 2> const& v) -> std::array<double, 2> {
            double r = 0.5 * (v[0] + 1.0);
            double phi = 0.5 * M_PI * v[1];
            // return {r * cos(phi), r * sin(phi)};
            return v;
        };
        std::array<double, 2> h = {1.0 / n, 1.0 / n};
        auto points = std::array<std::vector<double>, 2>{{{0.0, 0.75, 1.0}, {0.0, 0.5, 1.0}}};
        auto xbc = [](std::size_t plane, std::array<std::size_t, 1> const& regions) {
            if (plane == 1) {
                if (regions[0] == 1) {
                    return BC::Fault;
                }
                return BC::None;
            }
            return BC::Dirichlet;
        };
        auto ybc = [](std::size_t plane, std::array<std::size_t, 1> const&) {
            if (plane == 1) {
                return BC::None;
            }
            return BC::Dirichlet;
        };
        GenMesh<2> meshGen(points, h, {xbc, ybc});
        writeMesh(out, meshGen, transform, ghost);
    } else if (D == 3) {
        std::array<uint64_t, 3> N = {n, n, n};
        auto transform = [](std::array<double, 3> const& v) -> std::array<double, 3> {
            double r = 0.5 * (v[0] + 1.0);
            double phi = 0.5 * M_PI * v[1];
            double theta = M_PI * (0.5 * v[2] + 0.25);
            return {r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)};
        };
        std::array<std::pair<BC, BC>, 3> BCs;
        BCs.fill(std::make_pair(BC::Dirichlet, BC::Dirichlet));
        GenMesh<3> meshGen(N, BCs);
        writeMesh(out, meshGen, transform, ghost);
    } else {
        std::cerr << "Unsupported simplex dimension: " << D << std::endl;
    }

    MPI_Finalize();

    return 0;
}
