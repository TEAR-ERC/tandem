#include "poisson/Poisson.h"
#include "config.h"
#include "poisson/Scenario.h"
#include "writer.h"

#include "form/Error.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "mesh/GenMesh.h"
#include "mesh/MeshData.h"
#include "tensor/EigenMap.h"
#include "util/Hash.h"
#include "util/Stopwatch.h"

#include "xdmfwriter/XdmfWriter.h"
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
using xdmfwriter::TRIANGLE;
using xdmfwriter::XdmfWriter;

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
        auto ref = [](std::array<double, 2> const& x) { return exp(-x[0] - x[1] * x[1]); };
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
    case "analytic"_fnv1a: {
        auto ref = [](std::array<double, 2> const& x) { return exp(-x[0] - x[1] * x[1]); };
        return std::make_unique<MyScenario>(
            biunit, [](std::array<double, 2> const& x) { return 0.0; },
            [](std::array<double, 2> const& x) { return (x[0] < 0.0) ? 1.0 : 2.0; },
            [](Vector<double> const& x) -> std::array<double, 1> {
                return {(x(0) < 0.0) ? 1.0 : 2.0};
            });
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
                    MinQuadOrder());
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

    auto xt = poisson.reshapeNumericSolution(x);
    double error =
        tndm::Error<DomainDimension>::L2(poisson.refElement(), cl, xt, *scenario->reference());

    std::cout << "L2 error: " << error << std::endl;

    if (auto fileName = program.present("-o")) {
        std::vector<double> data(3 * mesh->numElements(), 0.0);
        auto outPoints = std::vector<std::array<double, 2u>>{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
        auto E = poisson.refElement().evaluateBasisAt(outPoints);
        auto EM = EigenMap(E);
        for (std::size_t elNo = 0; elNo < mesh->numElements(); ++elNo) {
            auto xtAtEl = xt.subtensor(tndm::slice{}, 0, elNo);
            Eigen::VectorXd out = EM.transpose() * EigenMap(xtAtEl);
            // Write nodes
            for (std::ptrdiff_t i = 0; i < 3; ++i) {
                data[3 * elNo + i] = out(i);
            }
            // data[elNo] = x(elNo * tndm::tensor::A::Shape[0]);
        }

        auto vertexData = dynamic_cast<VertexData<2u> const*>(mesh->vertices().data());
        if (!vertexData) {
            return 1;
        }

        // auto flatVerts = flatVertices<double, 2u, 3>(vertexData->getVertices(), transform);
        // auto flatElems = flatElements<unsigned int, 2u>(*mesh);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::vector<const char*> variableNames{"x"};
        XdmfWriter<TRIANGLE, double> writer(xdmfwriter::POSIX, (*fileName).c_str(), false);
        // writer.init(variableNames, std::vector<const char*>{});
        writer.init(std::vector<const char*>{}, variableNames);
        auto [cells, vertices] =
            duplicatedDofsMesh<unsigned int, double, 2u, 3u>(*mesh, scenario->transform());
        writer.setMesh(mesh->elements().size(), cells.data(), 3 * mesh->elements().size(),
                       vertices.data());
        // writer.setMesh(mesh->elements().size(), flatElems.data(),
        // vertexData->getVertices().size(), flatVerts.data());
        writer.addTimeStep(0.0);
        // writer.writeCellData(0, data.data());
        writer.writeVertexData(0, data.data());
        writer.flush();
        writer.close();
    }

    MPI_Finalize();

    return 0;
}
