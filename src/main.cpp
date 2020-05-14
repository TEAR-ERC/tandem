#include <iostream>

#include <mpi.h>

#include "mesh/Simplex.h"
#include "mesh/SimplexMesh.h"
#include "mesh/GenMesh.h"

#include "xdmfwriter/XdmfWriter.h"

using tndm::Simplex;
using tndm::SimplexMesh;
using tndm::generateUniformMesh;
using xdmfwriter::XdmfWriter;
using xdmfwriter::TRIANGLE;
using xdmfwriter::TETRAHEDRON;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    std::vector<SimplexMesh<2>::vertex_t> vertices;
    std::vector<Simplex<2>> elements;

    //std::array<int,2> N = {10,10};
    std::array<int,3> N = {128,128,128};
    auto mesh = generateUniformMesh(N);


    std::vector<const char*> variableNames{"x"};
    //XdmfWriter<TRIANGLE> writer(rank, "test", variableNames);
    XdmfWriter<TETRAHEDRON> writer(rank, "test", variableNames);
    auto flatVerts = mesh.flatVertices<double,3>();
    auto flatElems = mesh.flatElements<unsigned int>();
    writer.init(mesh.numElements(), flatElems.data(),
                mesh.numVertices(), flatVerts.data());
    writer.addTimeStep(0.0);

    MPI_Finalize();

    return 0;
}
