#include <iostream>

#include <mpi.h>
#include <parmetis.h>

#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/MeshData.h"

#include "xdmfwriter/XdmfWriter.h"

using tndm::BoundaryData;
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
    for (auto& elem : mesh.elements()) {
        auto lids = mesh.template downward<0>(elem);
        for (auto& lid : lids) {
            eout.push_back(static_cast<IntT>(lid));
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

    std::vector<const char*> variableNames{"x", "bc"};

    // constexpr std::size_t D = 2;
    // XdmfWriter<TRIANGLE> writer(rank, "testmesh", variableNames);
    // std::array<uint64_t, 2> N = {128, 128};

    constexpr std::size_t D = 3;
    XdmfWriter<TETRAHEDRON> writer(rank, "testmesh", variableNames);
    std::array<uint64_t, 3> N = {16, 16, 16};

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

    auto flatVerts = flatVertices<double, D, 3>(vertexData->getVertices());
    auto flatElems = flatElements<unsigned int, D>(*mesh);
    writer.init(mesh->elements().size(), flatElems.data(), vertexData->getVertices().size(),
                flatVerts.data());
    writer.addTimeStep(0.0);
    writer.writeData(0, data.data());
    writer.writeData(1, bc.data());
    writer.flush();

    MPI_Finalize();

    return 0;
}
