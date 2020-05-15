#include <iostream>

#include <mpi.h>
#include <parmetis.h>

#include "mesh/GlobalSimplexMesh.h"
#include "mesh/GenMesh.h"

#include "xdmfwriter/XdmfWriter.h"

using tndm::GenMesh;
using xdmfwriter::XdmfWriter;
using xdmfwriter::TRIANGLE;
using xdmfwriter::TETRAHEDRON;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    std::array<int,2> N = {128,128};
    auto globalMesh = GenMesh<2>::uniformMesh(N);
    //std::array<int,3> N = {128,128,128};
    //auto globalMesh = GenMesh<3>::uniformMesh(N);
    globalMesh.repartition();
    auto mesh = globalMesh.getLocalMesh();

    std::vector<const char*> variableNames{"x"};
    XdmfWriter<TRIANGLE> writer(rank, "test", variableNames);
    //XdmfWriter<TETRAHEDRON> writer(rank, "test", variableNames);
    auto flatVerts = mesh.flatVertices<double,3>();
    auto flatElems = mesh.flatElements<unsigned int>();
    writer.init(mesh.numElements(), flatElems.data(),
                mesh.numVertices(), flatVerts.data());
    writer.addTimeStep(0.0);

    MPI_Finalize();

    return 0;
}
