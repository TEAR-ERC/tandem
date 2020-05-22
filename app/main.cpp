#include <iostream>

#include <mpi.h>
#include <parmetis.h>

#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/MeshData.h"

#include "xdmfwriter/XdmfWriter.h"

using tndm::GenMesh;
using tndm::LocalSimplexMesh;
using tndm::VertexData;
using xdmfwriter::TETRAHEDRON;
using xdmfwriter::TRIANGLE;
using xdmfwriter::XdmfWriter;

template <typename IntT, std::size_t D>
std::vector<IntT> flatElements(LocalSimplexMesh<D> const& mesh) {
    std::vector<IntT> eout;
    eout.reserve((D + 1) * mesh.elements().size());
    for (auto& e : mesh.elements()) {
        for (auto& p : e) {
            eout.push_back(static_cast<IntT>(mesh.g2l(p)));
        }
    }
    return eout;
}
template <typename RealT, std::size_t D, std::size_t Dout = D>
std::vector<RealT> flatVertices(std::vector<std::array<double, D>> const& verts) {
    static_assert(Dout >= D);
    std::vector<RealT> vout;
    vout.reserve(Dout * verts.size());
    for (auto& v : verts) {
        std::size_t d;
        for (d = 0; d < D; ++d) {
            vout.push_back(static_cast<RealT>(v[d]));
        }
        for (; d < Dout; ++d) {
            vout.push_back(0.0);
        }
    }
    return vout;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    constexpr std::size_t D = 3;

    GenMesh<D> meshGen;
    // std::array<int,2> N = {128,128};
    // std::array<int, 3> N = {64, 64, 64};
    std::array<int, 3> N = {4, 4, 4};
    auto globalMesh = meshGen.uniformMesh(N);
    globalMesh.repartition();
    auto mesh = globalMesh.getLocalMesh();

    auto vertexData = dynamic_cast<VertexData<D> const*>(mesh.vertices().data());
    if (!vertexData) {
        return 1;
    }

    std::vector<double> data;
    data.reserve(mesh.elements().size());
    for (auto& e : mesh.elements()) {
        double numShared = 0.0;
        for (auto& p : e) {
            auto lid = mesh.g2l(p);
            numShared += mesh.vertices().getSharedRanks(lid).size();
        }
        data.push_back(numShared);
    }

    std::vector<const char*> variableNames{"x"};
    //XdmfWriter<TRIANGLE> writer(rank, "testmesh", variableNames);
    XdmfWriter<TETRAHEDRON> writer(rank, "testmesh", variableNames);
    auto flatVerts = flatVertices<double, D, 3>(vertexData->getVertices());
    auto flatElems = flatElements<unsigned int, D>(mesh);
    writer.init(mesh.elements().size(), flatElems.data(), vertexData->getVertices().size(),
                flatVerts.data());
    writer.addTimeStep(0.0);
    writer.writeData(0, data.data());
    writer.flush();

    MPI_Finalize();

    return 0;
}
