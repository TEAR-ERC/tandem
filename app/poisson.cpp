#include "config.h"
#include "writer.h"

#include "kernels/init.h"
#include "kernels/kernel.h"
#include "kernels/tensor.h"

#include "form/DG.h"
#include "geometry/Curvilinear.h"
#include "mesh/GenMesh.h"
#include "mesh/MeshData.h"

#include "xdmfwriter/XdmfWriter.h"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <argparse.hpp>
#include <mpi.h>

#include <cassert>
#include <iostream>

using tndm::VertexData;
using xdmfwriter::TRIANGLE;
using xdmfwriter::XdmfWriter;

namespace tndm {

class Poisson : DG<DomainDimension> {
public:
    using DG<DomainDimension>::DG;

    auto assemble() {
        using T = Eigen::Triplet<double>;
        std::vector<T> triplets;

        double D_x[tensor::D_x::size()];
        double A[tensor::A::size()];
        auto Aview = init::A::view::create(A);

        assert(volRule.size() == tensor::J::Shape[0]);
        assert(D_xi.shape(0) == tensor::D_xi::Shape[0]);
        assert(D_xi.shape(1) == tensor::D_xi::Shape[1]);
        assert(D_xi.shape(2) == tensor::D_xi::Shape[2]);

        for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
            kernel::assembleVolume krnl;
            krnl.A = A;
            krnl.D_x = D_x;
            krnl.D_xi = D_xi.data();
            krnl.G = vol[elNo].template get<JInvT>().data()->data();
            krnl.J = vol[elNo].template get<AbsDetJ>().data();
            krnl.W = volRule.weights().data();
            krnl.execute();
            for (unsigned i = 0; i < tensor::A::Shape[0]; ++i) {
                for (unsigned j = 0; j < tensor::A::Shape[1]; ++j) {
                    unsigned i0 = elNo * tensor::A::Shape[0];
                    unsigned j0 = elNo * tensor::A::Shape[1];
                    triplets.emplace_back(T(i0 + i, j0 + j, Aview(i, j)));
                }
            }
        }

        double a00[tensor::a::size(0, 0)];
        double a01[tensor::a::size(0, 1)];
        double a10[tensor::a::size(1, 0)];
        double a11[tensor::a::size(1, 1)];
        double d_x0[tensor::d_x::size(0)];
        double d_x1[tensor::d_x::size(1)];

        assert(d_xi[0].shape(0) == tensor::d_xi::Shape[0][0]);
        assert(d_xi[0].shape(1) == tensor::d_xi::Shape[0][1]);
        assert(d_xi[0].shape(2) == tensor::d_xi::Shape[0][2]);

        for (std::size_t fctNo = 0; fctNo < numFacets(); ++fctNo) {
            auto const& info = fctInfo[fctNo];
            kernel::assembleFacetLocal local;
            double half = (info.up[0] != info.up[1]) ? 0.5 : 1.0;
            local.c00 = -half;
            local.c10 = epsilon * half;
            local.c20 = penalty / info.area;
            local.a(0, 0) = a00;
            local.d_x(0) = d_x0;
            local.d_xi(0) = d_xi[info.localNo[0]].data();
            local.e(0) = e[info.localNo[0]].data();
            local.g = fct[fctNo].template get<JInvT>().data()->data();
            local.n = fct[fctNo].template get<Normal>().data()->data();
            local.nl = fct[fctNo].template get<NormalLength>().data();
            local.w = fctRule.weights().data();
            local.execute();

            auto push = [&info, &triplets](auto x, auto y, double* a) {
                auto aview = init::a::view<x(), y()>::create(a);
                for (unsigned i = 0; i < aview.shape(0); ++i) {
                    for (unsigned j = 0; j < aview.shape(1); ++j) {
                        unsigned i0 = info.up[x()] * aview.shape(0);
                        unsigned j0 = info.up[y()] * aview.shape(1);
                        triplets.emplace_back(T(i0 + i, j0 + j, aview(i, j)));
                    }
                }
            };

            push(std::integral_constant<int, 0>(), std::integral_constant<int, 0>(), a00);

            if (info.up[0] != info.up[1]) {
                kernel::assembleFacetNeighbour neighbour;
                neighbour.c00 = local.c00;
                neighbour.c01 = -local.c00;
                neighbour.c10 = local.c10;
                neighbour.c11 = -local.c10;
                neighbour.c20 = local.c20;
                neighbour.c21 = -local.c20;
                neighbour.a(0, 1) = a01;
                neighbour.a(1, 0) = a10;
                neighbour.a(1, 1) = a11;
                neighbour.d_x(0) = d_x0;
                neighbour.d_x(1) = d_x1;
                neighbour.d_xi(0) = local.d_xi(0);
                neighbour.d_xi(1) = d_xi[info.localNo[1]].data();
                neighbour.e(0) = local.e(0);
                neighbour.e(1) = e[info.localNo[1]].data();
                neighbour.g = local.g;
                neighbour.n = local.n;
                neighbour.nl = local.nl;
                neighbour.w = local.w;
                neighbour.execute();

                push(std::integral_constant<int, 0>(), std::integral_constant<int, 1>(), a01);
                push(std::integral_constant<int, 1>(), std::integral_constant<int, 0>(), a10);
                push(std::integral_constant<int, 1>(), std::integral_constant<int, 1>(), a11);
            }
        }

        Eigen::SparseMatrix<double> mat(numElements() * tensor::A::Shape[0],
                                        numElements() * tensor::A::Shape[1]);
        mat.setFromTriplets(triplets.begin(), triplets.end());
        return mat;
    }

    auto rhs() {
        Eigen::VectorXd B = Eigen::VectorXd::Zero(numElements() * tensor::A::Shape[0]);

        double b[tensor::b::size()];
        auto bview = init::b::view::create(b);

        double F[tensor::F::size()];
        assert(tensor::F::size() == volRule.size());
        for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
            auto coords = vol[elNo].template get<Coords>();
            for (unsigned q = 0; q < tensor::F::size(); ++q) {
                double y2 = coords[q][1] * coords[q][1];
                F[q] = (1.0 - 4.0 * y2) * exp(-coords[q][0] - y2);
            }

            kernel::rhsVolume rhs;
            rhs.E = E.data();
            rhs.F = F;
            rhs.J = vol[elNo].template get<AbsDetJ>().data();
            rhs.W = volRule.weights().data();
            rhs.b = b;
            rhs.execute();

            for (unsigned i = 0; i < bview.shape(0); ++i) {
                unsigned i0 = elNo * bview.shape(0);
                B[i0 + i] += bview(i);
            }
        }

        double f[tensor::f::size()];
        assert(tensor::f::size() == fctRule.size());

        for (std::size_t fctNo = 0; fctNo < numFacets(); ++fctNo) {
            auto const& info = fctInfo[fctNo];
            if (info.up[0] == info.up[1]) {
                auto coords = fct[fctNo].template get<Coords>();
                for (unsigned q = 0; q < tensor::f::size(); ++q) {
                    f[q] = exp(-coords[q][0] - coords[q][1] * coords[q][1]);
                }

                kernel::rhsFacet rhs;
                rhs.c10 = -epsilon;
                rhs.c20 = penalty / info.area;
                rhs.b = b;
                rhs.d_xi(0) = d_xi[info.localNo[0]].data();
                rhs.e(0) = e[info.localNo[0]].data();
                rhs.f = f;
                rhs.g = fct[fctNo].template get<JInvT>().data()->data();
                rhs.n = fct[fctNo].template get<Normal>().data()->data();
                rhs.nl = fct[fctNo].template get<NormalLength>().data();
                rhs.w = fctRule.weights().data();
                rhs.execute();

                for (unsigned i = 0; i < bview.shape(0); ++i) {
                    unsigned i0 = info.up[0] * bview.shape(0);
                    B[i0 + i] += bview(i);
                }
            }
        }
        return B;
    }

private:
    double epsilon = 1.0;
    double penalty = 1.0;
};

} // namespace tndm

using tndm::Curvilinear;
using tndm::GenMesh;
using tndm::Poisson;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    argparse::ArgumentParser program("poisson");
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

    auto n = program.get<unsigned long>("n");
    std::array<uint64_t, DomainDimension> size;
    size.fill(n);
    GenMesh meshGen(size);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();

    auto mesh = globalMesh->getLocalMesh();

    auto transform = [](std::array<double, 2> const& v) -> std::array<double, 2> {
        double r = 0.5 * (v[0] + 1.0);
        double phi = 0.5 * M_PI * v[1];
        return {r * cos(phi), r * sin(phi)};
    };

    Curvilinear<2u> cl(*mesh, transform, PolynomialDegree);

    Poisson poisson(*mesh, cl, PolynomialDegree, MinQuadOrder());
    auto A = poisson.assemble();
    auto b = poisson.rhs();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "LU of A failed" << std::endl;
        return -1;
    }
    auto x = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Could not solve Ax=b" << std::endl;
        return -1;
    }

    std::vector<double> data(mesh->numElements(), 0.0);
    for (std::size_t elNo = 0; elNo < mesh->numElements(); ++elNo) {
        // Write average
        data[elNo] = x(elNo * tndm::tensor::A::Shape[0]);
    }

    auto vertexData = dynamic_cast<VertexData<2u> const*>(mesh->vertices().data());
    if (!vertexData) {
        return 1;
    }

    auto flatVerts = flatVertices<double, 2u, 3>(vertexData->getVertices(), transform);
    auto flatElems = flatElements<unsigned int, 2u>(*mesh);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<const char*> variableNames{"x"};
    XdmfWriter<TRIANGLE> writer(rank, "poisson", variableNames);
    writer.init(mesh->elements().size(), flatElems.data(), vertexData->getVertices().size(),
                flatVerts.data());
    writer.addTimeStep(0.0);
    writer.writeData(0, data.data());
    writer.flush();

    MPI_Finalize();

    return 0;
}
