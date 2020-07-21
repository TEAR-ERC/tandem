#include "geometry/Curvilinear.h"
#include "geometry/Vector.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/LocalSimplexMesh.h"
#include "mesh/MeshData.h"
#include "quadrules/AutoRule.h"
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

using tndm::BC;
using tndm::Curvilinear;
using tndm::dot;
using tndm::GenMesh;
using tndm::LocalSimplexMesh;
using tndm::Managed;
using tndm::Matrix;
using tndm::simplexQuadratureRule;
using tndm::Tensor;
using tndm::Vector;

template <std::size_t... Is>
std::array<double, sizeof...(Is)> to_array(double* ptr, std::index_sequence<Is...>) {
    return {ptr[Is]...};
}

template <std::size_t D, typename SurfaceFunc>
double surfaceInt(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl, SurfaceFunc surfaceFun,
                  unsigned minQuadOrder) {
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

    double result = 0.0;
#pragma omp parallel shared(result)
    {
        auto J = Managed(cl.jacobianResultInfo(pts.size()));
        auto JInv = Managed(cl.jacobianResultInfo(pts.size()));
        auto detJ = Managed(cl.detJResultInfo(pts.size()));
        auto normal = Managed(cl.normalResultInfo(rule.size()));
        auto x1 = Managed(cl.mapResultInfo(rule.size()));
        auto x2 = Managed(cl.mapResultInfo(rule.size()));
        auto fx = Managed<Matrix<double>>(x1.shape());

#pragma omp for reduction(+ : result)
        for (std::size_t fNo = 0; fNo < mesh.numFacets(); ++fNo) {
            auto elNos = mesh.template upward<D - 1u>(fNo);
            assert(elNos.size() >= 1u);
            auto dws = mesh.template downward<D - 1u, D>(elNos[0]);
            auto localFNo = std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fNo));
            cl.jacobian(elNos[0], gradE[localFNo], J);
            cl.detJ(elNos[0], J, detJ);
            cl.jacobianInv(J, JInv);
            cl.map(elNos[0], E[localFNo], x1);
            cl.normal(localFNo, detJ, JInv, normal);
            for (std::ptrdiff_t q = 0; q < rule.size(); ++q) {
                auto f = surfaceFun(to_array(&x1(0, q), std::make_index_sequence<D>{}));
                std::copy(f.begin(), f.end(), &fx(0, q));
            }
            if (elNos.size() > 1) {
                auto dws2 = mesh.template downward<D - 1u, D>(elNos[1]);
                auto localFNo2 =
                    std::distance(dws2.begin(), std::find(dws2.begin(), dws2.end(), fNo));
                cl.map(elNos[1], E[localFNo2], x2);
                for (std::ptrdiff_t q = 0; q < rule.size(); ++q) {
                    auto f = surfaceFun(to_array(&x2(0, q), std::make_index_sequence<D>{}));
                    for (std::size_t d = 0; d < D; ++d) {
                        fx(d, q) -= f[d];
                    }
                }
            }
            for (std::ptrdiff_t q = 0; q < rule.size(); ++q) {
                for (std::ptrdiff_t d = 0; d < x1.shape(0); ++d) {
                    result += fx(d, q) * normal(d, q) * wgts[q];
                }
            }
        }
    }
    return result;
}

template <std::size_t D, typename VolumeFunc>
double volumeInt(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl, VolumeFunc volumeFun,
                 unsigned minQuadOrder) {
    auto rule = simplexQuadratureRule<D>(minQuadOrder);
    auto& pts = rule.points();
    auto& wgts = rule.weights();

    Managed<Matrix<double>> E = cl.evaluateBasisAt(pts);
    Managed<Tensor<double, 3u>> gradE = cl.evaluateGradientAt(pts);

    double result = 0.0;
#pragma omp parallel shared(result)
    {
        auto J = Managed(cl.jacobianResultInfo(pts.size()));
        auto JinvT = Managed(cl.jacobianResultInfo(pts.size()));
        auto absDetJ = Managed(cl.detJResultInfo(pts.size()));
        auto normal = Managed(cl.normalResultInfo(rule.size()));
        auto x1 = Managed(cl.mapResultInfo(rule.size()));
        auto x2 = Managed(cl.mapResultInfo(rule.size()));

#pragma omp for reduction(+ : result)
        for (std::size_t elNo = 0; elNo < mesh.numElements(); ++elNo) {
            cl.jacobian(elNo, gradE, J);
            cl.absDetJ(elNo, J, absDetJ);
            cl.map(elNo, E, x1);
            for (std::ptrdiff_t q = 0; q < rule.size(); ++q) {
                double fx = volumeFun(to_array(&x1(0, q), std::make_index_sequence<D>{}));
                result += fx * absDetJ(q) * wgts[q];
            }
        }
    }
    return result;
}

template <std::size_t D> auto getMesh(std::array<uint64_t, D> const& size) {
    std::array<std::pair<BC, BC>, D> BCs;
    BCs.fill(std::make_pair(BC::None, BC::None));
    GenMesh<D> meshGen(size, BCs);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();
    return globalMesh->getLocalMesh();
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
        auto surfaceFun = [](std::array<double, 2> const& x) -> std::array<double, 2> {
            double r = sqrt(x[0] * x[0] + x[1] * x[1]);
            double phi = atan2(x[1], x[0]);
            double f_r = sin(r) + cos(r) / r;
            return {f_r * cos(phi), f_r * sin(phi)};
        };
        auto volumeFun = [](std::array<double, 2> const& x) {
            double r = sqrt(x[0] * x[0] + x[1] * x[1]);
            return cos(r);
        };
        // int_{0}^{pi/2} int_{0.5}^{1} cos(r) r dr dphi
        reference = pi / 2.0 * (sin(1.0) + cos(1.0) - 0.5 * sin(0.5) - cos(0.5));
        testFun = [transform, surfaceFun, volumeFun, &N, &minQuadOrder](unsigned long n) {
            std::array<uint64_t, 2> size = {n, n};
            auto mesh = getMesh(size);
            Curvilinear<2> cl(*mesh, transform, N);
            return 0.5 * surfaceInt(*mesh, cl, surfaceFun, minQuadOrder) +
                   0.5 * volumeInt(*mesh, cl, volumeFun, minQuadOrder);
        };
    } else if (D == 3) {
        auto transform = [](std::array<double, 3> const& v) -> std::array<double, 3> {
            double r = 0.5 * (v[0] + 1.0);
            double phi = 0.5 * M_PI * v[1];
            double theta = M_PI * (0.5 * v[2] + 0.25);
            return {r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)};
        };
        auto surfaceFun = [](std::array<double, 3> const& x) -> std::array<double, 3> {
            return x * (1.0 / 3.0);
        };
        auto volumeFun = [](std::array<double, 3> const&) { return 1.0; };
        // int_{0}^{pi/2} int_{pi/4}^{3pi/4} int_{0.5}^{1} r^2 sin(theta) dr dtheta dphi
        reference = sqrt(2) * pi / 2.0 * 7.0 / 24.0;
        testFun = [transform, surfaceFun, volumeFun, &N, &minQuadOrder](unsigned long n) {
            std::array<uint64_t, 3> size = {n, n, n};
            auto mesh = getMesh(size);
            Curvilinear<3> cl(*mesh, transform, N);
            return 0.5 * surfaceInt(*mesh, cl, surfaceFun, minQuadOrder) +
                   0.5 * volumeInt(*mesh, cl, volumeFun, minQuadOrder);
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
