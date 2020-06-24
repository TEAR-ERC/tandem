#include "writer.h"

#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/MeshData.h"

#include "xdmfwriter/XdmfWriter.h"

#include <mpi.h>
#include <parmetis.h>

#include <cmath>
#include <iostream>

using tndm::BoundaryData;
using tndm::GenMesh;
using tndm::LocalSimplexMesh;
using tndm::VertexData;
using xdmfwriter::TETRAHEDRON;
using xdmfwriter::TRIANGLE;
using xdmfwriter::XdmfWriter;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    /*constexpr std::size_t D = 2;
    XdmfWriter<TRIANGLE> writer(rank, "testmesh", variableNames);
    std::array<uint64_t, 2> N = {32, 32};
    auto transform = [](std::array<double, 2> const& v) -> std::array<double, 2> {
        double r = 0.5 * (v[0] + 1.0);
        double phi = 0.5 * M_PI * v[1];
        return {r * cos(phi), r * sin(phi)};
    };*/

    constexpr std::size_t D = 3;
    XdmfWriter<TETRAHEDRON, double> writer(xdmfwriter::POSIX, "testmesh");
    std::array<uint64_t, 3> N = {32, 32, 32};
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

    auto vertexData = dynamic_cast<VertexData<D> const*>(mesh->vertices().data());
    auto boundaryData = dynamic_cast<BoundaryData const*>(mesh->facets().data());
    if (!vertexData || !boundaryData) {
        return 1;
    }

    std::vector<double> data(mesh->numElements(), 0.0);
    std::vector<double> bc(mesh->numElements(), 0.0);
    for (std::size_t fid = 0; fid < mesh->numFacets(); ++fid) {
        auto& eids = mesh->upward<D - 1>(fid);
        for (auto& eid : eids) {
            assert(eid < data.size());
            data[eid] += mesh->facets().getSharedRanks(fid).size();
            bc[eid] += boundaryData->getBoundaryConditions()[fid];
        }
    }

    auto flatVerts = flatVertices<double, D, 3>(vertexData->getVertices(), transform);
    auto flatElems = flatElements<unsigned int, D>(*mesh);

    std::vector<const char*> variableNames{"x", "bc"};
    writer.init(variableNames, std::vector<char const*>{});
    writer.setMesh(mesh->elements().size(), flatElems.data(), vertexData->getVertices().size(),
                   flatVerts.data());
    writer.addTimeStep(0.0);
    writer.writeCellData(0, data.data());
    writer.writeCellData(1, bc.data());
    writer.flush();
    writer.close();

    MPI_Finalize();

    return 0;
}
