#include "basis/Quadrature.h"
#include "geometry/Curvilinear.h"
#include "geometry/Vector.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "tensor/Tensor.h"

#include <argparse.hpp>
#include <mpi.h>

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

using tndm::createSimplexQuadratureRule;
using tndm::Curvilinear;
using tndm::dot;
using tndm::GenMesh;
using tndm::Tensor;

template <std::size_t D, typename Func>
double test(std::array<uint64_t, D> const& size, Func transform, unsigned degree) {
    GenMesh<D> meshGen(size);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();

    auto mesh = globalMesh->getLocalMesh();

    Curvilinear<D> cl(*mesh, transform, degree);

    // auto rule = createSimplexQuadratureRule<D - 1u>(degree + 1);
    auto rule = createSimplexQuadratureRule<D>(degree + 1);
    auto& pts = rule.points();
    auto& wgts = rule.weights();

    auto gradE = cl.evaluateGradientAt(pts);
    auto J = Tensor(cl.jacobianResultInfo(pts.size()));
    auto Jview = J.view();
    auto detJ = Tensor(cl.detJResultInfo(pts.size()));
    auto detJview = detJ.view();

    double volume = 0.0;
    for (std::size_t eleNo = 0; eleNo < mesh->numElements(); ++eleNo) {
        double localSum = 0.0;

        cl.jacobian(eleNo, gradE, Jview);
        cl.detJ(eleNo, Jview, detJview);
        for (std::size_t q = 0; q < rule.size(); ++q) {
            localSum += std::fabs(detJ(q)) * wgts[q];
        }
        volume += localSum;

        // for (std::size_t f = 0; f < D + 1; ++f) {
        for (std::size_t q = 0; q < rule.size(); ++q) {
            // auto xi = cl.facetParam(f, pts[q]);
            // localSum += dot(cl.map(eleNo, xi), cl.normal(eleNo, f, xi)) * wgts[q];
        }
        //}
        // volume += localSum / D;
    }

    return volume;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    argparse::ArgumentParser program("test-gauss");
    program.add_argument("-D", "--dim")
        .help("Simplex dimension (D=2: triangle, D=3: tet)")
        .default_value(2ul)
        .action([](std::string const& value) { return std::stoul(value); });
    program.add_argument("-N", "--degree")
        .help("Polynomial degree for geometry approximation")
        .default_value(1ul)
        .action([](std::string const& value) { return std::stoul(value); });
    program.add_argument("--csv")
        .help("Print in CSV format")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--csv-headless")
        .help("Print in CSV without header")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("n")
        .help("Number of elements per dimension")
        .remaining()
        .action([](std::string const& value) { return std::stoul(value); });

    try {
        program.parse_args(argc, argv);
    } catch (std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 0;
    }

    auto D = program.get<unsigned long>("-D");
    auto N = program.get<unsigned long>("-N");
    auto csv = program.get<bool>("--csv");
    auto csvHeadless = program.get<bool>("--csv-headless");
    auto n = program.get<std::vector<unsigned long>>("n");

    constexpr double pi = 3.1415926535897932384626433832795028841971693993751058;

    double reference = 0.0;
    std::function<double(unsigned long)> testFun;
    if (D == 2) {
        auto transform = [](Curvilinear<2>::vertex_t const& v) {
            double x = 2.0 * v[0] - 1.0;
            double y = 2.0 * v[1] - 1.0;
            return Curvilinear<2>::vertex_t{x * sqrt(1.0 - y * y / 2.0),
                                            y * sqrt(1.0 - x * x / 2.0)};
        };
        reference = pi;
        testFun = [&transform, &N](unsigned long n) {
            std::array<uint64_t, 2> size = {n, n};
            return test(size, transform, N);
        };
    } else if (D == 3) {
        auto transform = [](Curvilinear<3>::vertex_t const& v) {
            double x = 2.0 * v[0] - 1.0;
            double y = 2.0 * v[1] - 1.0;
            double z = 2.0 * v[2] - 1.0;
            return Curvilinear<3>::vertex_t{
                x * sqrt(1.0 - y * y / 2.0 - z * z / 2.0 + y * y * z * z / 3.0),
                y * sqrt(1.0 - x * x / 2.0 - z * z / 2.0 + x * x * z * z / 3.0),
                z * sqrt(1.0 - x * x / 2.0 - y * y / 2.0 + x * x * y * y / 3.0)};
        };
        reference = 4.0 * pi / 3.0;
        testFun = [&transform, &N](unsigned long n) {
            std::array<uint64_t, 3> size = {n, n, n};
            return test(size, transform, N);
        };
    } else {
        std::cerr << "Test for simplex dimension " << D << " is not implemented." << std::endl;
        return -1;
    }

    if (csv && !csvHeadless) {
        std::cout << "D,degree,n,computed,error" << std::endl;
    }
    std::cout << std::setprecision(16);

    for (auto& nn : n) {
        double volumeLocal = testFun(nn);
        double volume;
        MPI_Reduce(&volumeLocal, &volume, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            double error = std::fabs(reference - volume);
            if (csv || csvHeadless) {
                std::cout << D << "," << N << "," << nn << "," << volume << "," << error
                          << std::endl;
            } else {
                std::cout << "n = " << nn << ":" << std::endl;
                std::cout << "\tReference area: " << reference << std::endl;
                std::cout << "\tComputed area: " << volume << std::endl;
                std::cout << "\tError: " << error << std::endl;
            }
        }
    }

    MPI_Finalize();

    return 0;
}
