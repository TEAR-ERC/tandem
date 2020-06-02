#include "basis/Quadrature.h"
#include "geometry/Curvilinear.h"
#include "geometry/Vector.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"

#include <mpi.h>

#include <cmath>
#include <iostream>

using tndm::Curvilinear;
using tndm::dot;
using tndm::GenMesh;
using tndm::IntervalQuadrature;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    if (argc < 2) {
        std::cerr << "Usage: test-gauss <n>" << std::endl;
        return -1;
    }
    uint64_t n = atoi(argv[1]);

    constexpr std::size_t D = 2;
    std::array<uint64_t, 2> N = {n, n};
    auto transform = [](Curvilinear<D>::vertex_t const& v) {
        double x = 2.0 * v[0] - 1.0;
        double y = 2.0 * v[1] - 1.0;
        return Curvilinear<D>::vertex_t{x * sqrt(1.0 - y * y / 2.0), y * sqrt(1.0 - x * x / 2.0)};
    };

    // constexpr std::size_t D = 3;
    // std::array<uint64_t, 3> N = {8, 8, 8};
    // auto transform = [](GenMesh<D>::vertex_t const& v) {
    // return v;
    //};

    GenMesh<D> meshGen(N);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();

    auto mesh = globalMesh->getLocalMesh();

    Curvilinear<D> cl(*mesh, transform, true);

    auto rule = IntervalQuadrature(2);
    auto& pts = rule.points();
    auto& wgts = rule.weights();

    double volume = 0.0;
    for (std::size_t eleNo = 0; eleNo < mesh->numElements(); ++eleNo) {
        double localSum = 0.0;
        for (std::size_t f = 0; f < D + 1; ++f) {
            for (std::size_t q = 0; q < rule.size(); ++q) {
                auto xi = cl.facetParam(f, pts[q]);
                localSum += dot(cl.map(eleNo, xi), cl.normal(eleNo, f, xi)) * wgts[q];
            }
        }
        volume += localSum / D;
    }

    double pi = 3.14159265359;

    std::cout << "Reference area: " << pi << std::endl;
    std::cout << "Computed area: " << volume << std::endl;
    std::cout << "Error: " << std::fabs(pi - volume) << std::endl;

    MPI_Finalize();

    return 0;
}
