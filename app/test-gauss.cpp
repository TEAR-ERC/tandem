#include "quadrules/AutoRule.h"
#include "geometry/Curvilinear.h"
#include "geometry/Vector.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "tensor/Tensor.h"

#include <argparse.hpp>
#include <cstddef>
#include <limits>
#include <mpi.h>

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

using tndm::simplexQuadratureRule;
using tndm::Curvilinear;
using tndm::dot;
using tndm::GenMesh;
using tndm::Managed;
using tndm::Matrix;
using tndm::Tensor;

template <std::size_t D, typename Func>
double test(std::array<uint64_t, D> const& size, Func transform, unsigned degree,
            unsigned minQuadOrder) {
    GenMesh<D> meshGen(size);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();

    auto mesh = globalMesh->getLocalMesh();

    Curvilinear<D> cl(*mesh, transform, degree);

    auto rule = simplexQuadratureRule<D - 1u>(minQuadOrder);
    auto& pts = rule.points();
    auto& wgts = rule.weights();

    std::vector<Managed<Matrix<double>>> E;
    for (std::size_t f = 0; f < D + 1; ++f) {
        E.emplace_back(cl.evaluateBasisAt(cl.facetParam(f, pts)));
    }

    std::vector<Managed<Tensor<double, 3u>>> gradE;
    for (std::size_t f = 0; f < D + 1; ++f) {
        gradE.emplace_back(cl.evaluateGradientAt(cl.facetParam(f, pts)));
    }

    double volume = 0.0;
#pragma omp parallel shared(volume)
    {
        auto J = Managed(cl.jacobianResultInfo(pts.size()));
        auto JinvT = Managed(cl.jacobianResultInfo(pts.size()));
        auto detJ = Managed(cl.detJResultInfo(pts.size()));
        auto normal = Managed(cl.normalResultInfo(rule.size()));
        auto x1 = Managed(cl.mapResultInfo(rule.size()));
        auto x2 = Managed(cl.mapResultInfo(rule.size()));

#pragma omp for reduction(+ : volume)
        for (std::size_t fNo = 0; fNo < mesh->numFacets(); ++fNo) {
            auto elNos = mesh->template upward<D - 1u>(fNo);
            assert(elNos.size() >= 1u);
            auto dws = mesh->template downward<D - 1u, D>(elNos[0]);
            auto localFNo = std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fNo));
            cl.jacobian(elNos[0], gradE[localFNo], J);
            cl.detJ(elNos[0], J, detJ);
            cl.jacobianInvT(J, JinvT);
            cl.map(elNos[0], E[localFNo], x1);
            cl.normal(localFNo, detJ, JinvT, normal);
            if (elNos.size() > 1) {
                auto dws2 = mesh->template downward<D - 1u, D>(elNos[1]);
                auto localFNo2 = std::distance(dws2.begin(), std::find(dws2.begin(), dws2.end(), fNo));
                cl.map(elNos[1], E[localFNo2], x2);
                for (std::ptrdiff_t q = 0; q < rule.size(); ++q) {
                    for (std::ptrdiff_t d = 0; d < x1.shape(0); ++d) {
                        x1(d, q) -= x2(d, q);
                    }
                }
            }
            double localSum = 0.0;
            for (std::ptrdiff_t q = 0; q < rule.size(); ++q) {
                for (std::ptrdiff_t d = 0; d < x1.shape(0); ++d) {
                    localSum += x1(d, q) * normal(d, q) * wgts[q];
                }
            }
            volume += localSum / D;
        } 
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
    program.add_argument("-Q", "--min-quadrature-order")
        .help("Minimum quadrature order")
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
    unsigned long minQuadOrder = 2u * N + 1u;
    if (auto Q = program.present<unsigned long>("-Q")) {
        minQuadOrder = *Q;
    }
    auto csv = program.get<bool>("--csv");
    auto csvHeadless = program.get<bool>("--csv-headless");
    auto n = program.get<std::vector<unsigned long>>("n");

    constexpr double pi = 3.1415926535897932384626433832795028841971693993751058;

    double reference = 0.0;
    std::function<double(unsigned long)> testFun;
    if (D == 2) {
        auto transform = [](std::array<double, 2> const& v) -> std::array<double, 2> {
            double r = 0.5 * (v[0] + 1.0);
            double phi = 0.5 * M_PI * v[1];
            return {r * cos(phi), r * sin(phi)};
        };
        reference = 3.0 * pi / 16.0;
        testFun = [&transform, &N, &minQuadOrder](unsigned long n) {
            std::array<uint64_t, 2> size = {n, n};
            return test(size, transform, N, minQuadOrder);
        };
    } else if (D == 3) {
        auto transform = [](std::array<double, 3> const& v) -> std::array<double, 3> {
            double r = 0.5 * (v[0] + 1.0);
            double phi = 0.5 * M_PI * v[1];
            double theta = M_PI * (0.5 * v[2] + 0.25);
            return {r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)};
        };
        reference = sqrt(2) * pi / 2.0 * 7.0 / 24.0;
        testFun = [&transform, &N, &minQuadOrder](unsigned long n) {
            std::array<uint64_t, 3> size = {n, n, n};
            return test(size, transform, N, minQuadOrder);
        };
    } else {
        std::cerr << "Test for simplex dimension " << D << " is not implemented." << std::endl;
        return -1;
    }

    if (csv && !csvHeadless) {
        std::cout << "D,degree,minQuadOrder,n,computed,error" << std::endl;
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
                std::cout << D << "," << N << "," << minQuadOrder << "," << nn << "," << volume
                          << "," << error << std::endl;
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
