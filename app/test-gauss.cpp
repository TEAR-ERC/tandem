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
using tndm::TriangleQuadrature;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    if (argc < 2) {
        std::cerr << "Usage: test-gauss <n>" << std::endl;
        return -1;
    }
    uint64_t n = atoi(argv[1]);

    constexpr double pi = 3.1415926535897932384626433832795028841971693993751058;

    constexpr std::size_t D = 2;
    std::array<uint64_t, 2> N = {n, n};
    auto transform = [](Curvilinear<D>::vertex_t const& v) {
        double x = 2.0 * v[0] - 1.0;
        double y = 2.0 * v[1] - 1.0;
        return Curvilinear<D>::vertex_t{x * sqrt(1.0 - y * y / 2.0), y * sqrt(1.0 - x * x / 2.0)};
    };
    unsigned degree = 4;
    auto rule = IntervalQuadrature(degree);
    auto reference = pi;

    // constexpr std::size_t D = 3;
    // std::array<uint64_t, 3> N = {n, n, n};
    // auto transform = [](Curvilinear<D>::vertex_t const& v) {
    // double x = 2.0 * v[0] - 1.0;
    // double y = 2.0 * v[1] - 1.0;
    // double z = 2.0 * v[2] - 1.0;
    // return Curvilinear<D>::vertex_t{
    // x * sqrt(1.0 - y * y / 2.0 - z * z / 2.0 + y * y * z * z / 3.0),
    // y * sqrt(1.0 - x * x / 2.0 - z * z / 2.0 + x * x * z * z / 3.0),
    // z * sqrt(1.0 - x * x / 2.0 - y * y / 2.0 + x * x * y * y / 3.0)};
    //};
    // auto rule = TriangleQuadrature(2);
    // auto reference = 4.0 * pi / 3.0;

    GenMesh<D> meshGen(N);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();

    auto mesh = globalMesh->getLocalMesh();

    Curvilinear<D> cl(*mesh, transform, degree);
    // Curvilinear<D> cl(*mesh, transform);

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

    std::cout << "Reference area: " << reference << std::endl;
    std::cout << "Computed area: " << volume << std::endl;
    std::cout << "Error: " << std::fabs(reference - volume) << std::endl;

    MPI_Finalize();

    return 0;
}
