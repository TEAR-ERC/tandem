#include "geometry/Curvilinear.h"
#include "io/VTUWriter.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/MeshData.h"

#include <mpi.h>
#include <parmetis.h>

#include <cmath>
#include <iostream>

using tndm::BoundaryData;
using tndm::Curvilinear;
using tndm::GenMesh;
using tndm::LocalSimplexMesh;
using tndm::VertexData;
using tndm::VTUWriter;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    /*constexpr std::size_t D = 2;
    std::array<uint64_t, 2> N = {1, 1};
    auto transform = [](std::array<double, 2> const& v) -> std::array<double, 2> {
        double r = 0.5 * (v[0] + 1.0);
        double phi = 0.5 * M_PI * v[1];
        return {r * cos(phi), r * sin(phi)};
    };*/

    constexpr std::size_t D = 3;
    std::array<uint64_t, 3> N = {1, 1, 1};
    auto transform = [](std::array<double, 3> const& v) -> std::array<double, 3> {
        double r = 0.5 * (v[0] + 1.0);
        double phi = 0.5 * M_PI * v[1];
        double theta = M_PI * (0.5 * v[2] + 0.25);
        return {r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)};
    };

    GenMesh<D> meshGen(N);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();

    auto mesh = globalMesh->getLocalMesh();

    unsigned degree = 15;
    Curvilinear<D> cl(*mesh, *transform, degree);

    auto boundaryData = dynamic_cast<BoundaryData const*>(mesh->facets().data());
    if (!boundaryData) {
        return -1;
    }
    std::vector<double> bc(mesh->numElements(), 0.0);
    for (std::size_t fid = 0; fid < mesh->numFacets(); ++fid) {
        auto& eids = mesh->upward<D - 1>(fid);
        for (auto& eid : eids) {
            bc[eid] += boundaryData->getBoundaryConditions()[fid];
        }
    }

    VTUWriter<D> writer(degree);
    auto piece = writer.addPiece(cl);
    piece.addCellData("BC", bc);
    writer.write("testmesh");

    MPI_Finalize();

    return 0;
}
