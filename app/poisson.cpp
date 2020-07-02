#include "poisson/Poisson.h"
#include "config.h"
#include "poisson/Scenario.h"

#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "io/VTUWriter.h"
#include "mesh/GenMesh.h"
#include "mesh/MeshData.h"
#include "tensor/Tensor.h"
#include "util/Hash.h"
#include "util/Stopwatch.h"

#include <Eigen/Core>
#include <Eigen/SparseLU>
#include <argparse.hpp>
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <iostream>
#include <memory>
#include <tuple>

using tndm::Curvilinear;
using tndm::fnv1a;
using tndm::GenMesh;
using tndm::operator""_fnv1a;
using tndm::MyScenario;
using tndm::Poisson;
using tndm::Scenario;
using tndm::Vector;
using tndm::VertexData;
using tndm::VTUWriter;

std::unique_ptr<Scenario> getScenario(std::string const& name) {
    auto partialAnnulus = [](std::array<double, 2> const& v) -> std::array<double, 2> {
        double r = 0.5 * (v[0] + 1.0);
        double phi = 0.5 * M_PI * v[1];
        return {r * cos(phi), r * sin(phi)};
    };
    auto biunit = [](std::array<double, 2> const& v) -> std::array<double, 2> {
        return {2.0 * v[0] - 1.0, 2.0 * v[1] - 1.0};
    };
    switch (tndm::fnv1a(name)) {
    case "manufactured"_fnv1a: {
        return std::make_unique<MyScenario>(
            partialAnnulus,
            [](std::array<double, 2> const& x) {
                return (1.0 - 4.0 * x[1] * x[1]) * exp(-x[0] - x[1] * x[1]);
            },
            [](std::array<double, 2> const& x) { return exp(-x[0] - x[1] * x[1]); },
            [](Vector<double> const& x) -> std::array<double, 1> {
                return {exp(-x(0) - x(1) * x(1))};
            });
    }
    case "manufactured_variable"_fnv1a: {
        return std::make_unique<MyScenario>(
            partialAnnulus,
            [](std::array<double, 2> const& x) {
                return (1.0 + 3.0 * x[1] - 4.0 * x[1] * x[1] * x[1] + x[0] -
                        4.0 * x[0] * x[1] * x[1]) *
                       exp(-x[0] - x[1] * x[1]);
            },
            [](std::array<double, 2> const& x) { return exp(-x[0] - x[1] * x[1]); },
            [](Vector<double> const& x) -> std::array<double, 1> {
                return {exp(-x(0) - x(1) * x(1))};
            },
            [](std::array<double, 2> const& x) { return x[0] + x[1]; });
    }
    case "cosine"_fnv1a: {
        double f = 10.0;
        auto ref1D = [f](double x) { return cos(f * M_PI * x); };
        return std::make_unique<MyScenario>(
            partialAnnulus,
            [f, ref1D](std::array<double, 2> const& x) {
                return 2.0 * f * f * M_PI * M_PI * ref1D(x[0]) * ref1D(x[1]);
            },
            [ref1D](std::array<double, 2> const& x) { return ref1D(x[0]) * ref1D(x[1]); },
            [ref1D](Vector<double> const& x) -> std::array<double, 1> {
                return {ref1D(x(0)) * ref1D(x(1))};
            });
    }
    case "singular"_fnv1a: {
        auto sol = [](std::array<double, 2> const& x) {
            double r = hypot(x[0], x[1]);
            double phi = atan2(x[1], x[0]);
            if (phi < 0) {
                phi += 2.0 * M_PI;
            }
            double const delta = 0.5354409456;
            std::array<double, 4> const a{0.4472135955, -0.7453559925, -0.9441175905, -2.401702643};
            std::array<double, 4> const b{1.0, 2.333333333, 0.55555555555, -0.4814814814};
            int dNo = 0;
            if (x[0] < 0 && x[1] > 0) {
                dNo = 1;
            } else if (x[0] < 0 && x[1] < 0) {
                dNo = 2;
            } else if (x[0] > 0 && x[1] < 0) {
                dNo = 3;
            }
            return std::pow(r, delta) * (a[dNo] * sin(delta * phi) + b[dNo] * cos(delta * phi));
        };
        return std::make_unique<MyScenario>(
            biunit, [](std::array<double, 2> const& x) { return 0.0; },
            [sol](std::array<double, 2> const& x) { return sol(x); },
            [sol](Vector<double> const& x) -> std::array<double, 1> {
                return {sol({x(0), x(1)})};
            },
            [](std::array<double, 2> const& x) { return (x[0] * x[1] >= 0) ? 5.0 : 1.0; });
    }
    default:
        return nullptr;
    }
    return nullptr;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    argparse::ArgumentParser program("poisson");
    program.add_argument("-o").help("Output file name");
    program.add_argument("-s")
        .default_value(std::string("manufactured"))
        .help("Scenario name")
        .action([](std::string const& value) {
            std::string result;
            std::transform(value.begin(), value.end(), std::back_inserter(result),
                           [](unsigned char c) { return std::tolower(c); });
            return result;
        });
    program.add_argument("n")
        .help("Number of elements per dimension")
        .action([](std::string const& value) { return std::stoul(value); });

    try {
        program.parse_args(argc, argv);
    } catch (std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 0;
    }

    auto scenario = getScenario(program.get("-s"));
    if (!scenario) {
        std::cerr << "Unknown scenario " << program.get("-s") << std::endl;
        return -1;
    }

    auto n = program.get<unsigned long>("n");
    std::array<uint64_t, DomainDimension> size;
    size.fill(n);
    GenMesh meshGen(size);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();

    auto mesh = globalMesh->getLocalMesh();

    Curvilinear<DomainDimension> cl(*mesh, scenario->transform(), PolynomialDegree);

    tndm::Stopwatch sw;

    sw.start();
    Poisson poisson(*mesh, cl, std::make_unique<tndm::ModalRefElement<2ul>>(PolynomialDegree),
                    MinQuadOrder(), scenario->K());
    std::cout << "Constructed Poisson after " << sw.split() << std::endl;

    auto A = poisson.assemble();
    auto b = poisson.rhs(scenario->force(), scenario->dirichlet());
    std::cout << "Assembled after " << sw.split() << std::endl;

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    std::cout << "LU after " << sw.split() << std::endl;
    if (solver.info() != Eigen::Success) {
        std::cerr << "LU of A failed" << std::endl;
        return -1;
    }
    Eigen::VectorXd x = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Could not solve Ax=b" << std::endl;
        return -1;
    }

    auto numeric = poisson.finiteElementFunction(x);
    double error = tndm::Error<DomainDimension>::L2(cl, numeric, *scenario->reference());
    std::cout << "L2 error: " << error << std::endl;

    if (auto fileName = program.present("-o")) {
        VTUWriter<2u> writer(PolynomialDegree);
        auto piece = writer.addPiece(cl);
        piece.addPointData("u", numeric);
        writer.write(*fileName);
    }

    MPI_Finalize();

    return 0;
}
