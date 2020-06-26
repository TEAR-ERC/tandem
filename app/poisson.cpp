#include "config.h"
#include "writer.h"

#include "kernels/init.h"
#include "kernels/kernel.h"
#include "kernels/tensor.h"

#include "form/DG.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "mesh/GenMesh.h"
#include "mesh/MeshData.h"
#include "tensor/EigenMap.h"
#include "tensor/Tensor.h"
#include "util/Stopwatch.h"

#include "xdmfwriter/XdmfWriter.h"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <argparse.hpp>
#include <mpi.h>

#include <cassert>
#include <cmath>
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

        assert(volRule.size() == tensor::W::Shape[0]);
        assert(D_xi.shape(0) == tensor::D_xi::Shape[0]);
        assert(D_xi.shape(1) == tensor::D_xi::Shape[1]);
        assert(D_xi.shape(2) == tensor::D_xi::Shape[2]);

        for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
            kernel::assembleVolume krnl;
            krnl.A = A;
            krnl.D_x = D_x;
            krnl.D_xi = D_xi.data();
            krnl.G = vol[elNo].template get<JInv>().data()->data();
            krnl.J = vol[elNo].template get<AbsDetJ>().data();
            krnl.W = volRule.weights().data();
            krnl.execute();
            unsigned i0 = elNo * tensor::A::Shape[0];
            unsigned j0 = elNo * tensor::A::Shape[1];
            for (unsigned i = 0; i < tensor::A::Shape[0]; ++i) {
                for (unsigned j = 0; j < tensor::A::Shape[1]; ++j) {
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

        assert(fctRule.size() == tensor::w::Shape[0]);
        assert(e[0].shape(0) == tensor::e::Shape[0][0]);
        assert(e[0].shape(1) == tensor::e::Shape[0][1]);
        assert(d_xi[0].shape(0) == tensor::d_xi::Shape[0][0]);
        assert(d_xi[0].shape(1) == tensor::d_xi::Shape[0][1]);
        assert(d_xi[0].shape(2) == tensor::d_xi::Shape[0][2]);

        for (std::size_t fctNo = 0; fctNo < numFacets(); ++fctNo) {
            auto const& info = fctInfo[fctNo];
            kernel::assembleFacetLocal local;
            double half = (info.up[0] != info.up[1]) ? 0.5 : 1.0;
            local.c00 = -half;
            local.c10 = epsilon * half;
            local.c20 = penalty / std::pow(info.area, beta0);
            local.a(0, 0) = a00;
            local.d_x(0) = d_x0;
            local.d_xi(0) = d_xi[info.localNo[0]].data();
            local.e(0) = e[info.localNo[0]].data();
            local.g = fct[fctNo].template get<JInv>().data()->data();
            local.n = fct[fctNo].template get<Normal>().data()->data();
            local.nl = fct[fctNo].template get<NormalLength>().data();
            local.w = fctRule.weights().data();
            local.execute();

            auto push = [&info, &triplets](auto x, auto y, double* a) {
                auto aview = init::a::view<x(), y()>::create(a);
                unsigned i0 = info.up[x()] * aview.shape(0);
                unsigned j0 = info.up[y()] * aview.shape(1);
                for (unsigned i = 0; i < aview.shape(0); ++i) {
                    for (unsigned j = 0; j < aview.shape(1); ++j) {
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
                neighbour.d_xi(1) = d_xi[info.localNo[1]].data();
                neighbour.e(0) = local.e(0);
                neighbour.e(1) = e[info.localNo[1]].data();
                neighbour.g = fct[fctNo].template get<JInvOther>().data()->data();
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

    template <typename ForceFunc, typename DirichletFunc>
    auto rhs(ForceFunc forceFun, DirichletFunc dirichletFun) {
        Eigen::VectorXd B = Eigen::VectorXd::Zero(numElements() * tensor::A::Shape[0]);

        double b[tensor::b::size()];
        auto bview = init::b::view::create(b);

        assert(tensor::b::Shape[0] == tensor::A::Shape[0]);

        double F[tensor::F::size()];
        assert(tensor::F::size() == volRule.size());
        for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
            auto coords = vol[elNo].template get<Coords>();
            for (unsigned q = 0; q < tensor::F::size(); ++q) {
                F[q] = forceFun(coords[q]);
            }

            kernel::rhsVolume rhs;
            rhs.E = E.data();
            rhs.F = F;
            rhs.J = vol[elNo].template get<AbsDetJ>().data();
            rhs.W = volRule.weights().data();
            rhs.b = b;
            rhs.execute();

            unsigned i0 = elNo * bview.shape(0);
            for (unsigned i = 0; i < bview.shape(0); ++i) {
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
                    f[q] = dirichletFun(coords[q]);
                }

                kernel::rhsFacet rhs;
                rhs.c10 = epsilon;
                rhs.c20 = penalty / std::pow(info.area, beta0);
                rhs.b = b;
                rhs.d_xi(0) = d_xi[info.localNo[0]].data();
                rhs.e(0) = e[info.localNo[0]].data();
                rhs.f = f;
                rhs.g = fct[fctNo].template get<JInv>().data()->data();
                rhs.n = fct[fctNo].template get<Normal>().data()->data();
                rhs.nl = fct[fctNo].template get<NormalLength>().data();
                rhs.w = fctRule.weights().data();
                rhs.execute();

                unsigned i0 = info.up[0] * bview.shape(0);
                for (unsigned i = 0; i < bview.shape(0); ++i) {
                    B[i0 + i] += bview(i);
                }
            }
        }
        return B;
    }

private:
    double epsilon = -1.0;
    double penalty = (PolynomialDegree + 1) * (PolynomialDegree + DomainDimension) /
                     DomainDimension * 6.8284271247461900976;
    double beta0 = 1.0;
};

template <std::size_t D, typename Func>
double L2error(Curvilinear<D>& cl, Eigen::VectorXd const& numeric, Func reference) {
    auto rule = simplexQuadratureRule<D>(20);

    auto E = dubinerBasisAt(PolynomialDegree, rule.points());
    auto geoE = cl.evaluateBasisAt(rule.points());
    auto geoD_xi = cl.evaluateGradientAt(rule.points());
    auto J = Managed(cl.jacobianResultInfo(rule.size()));
    auto absDetJ = Managed(cl.detJResultInfo(rule.size()));
    auto coords = Managed(cl.mapResultInfo(rule.size()));

    Eigen::VectorXd x(rule.size());

    double error = 0.0;
    for (std::size_t elNo = 0; elNo < cl.numElements(); ++elNo) {
        cl.jacobian(elNo, geoD_xi, J);
        cl.absDetJ(elNo, J, absDetJ);
        cl.map(elNo, geoE, coords);
        // int (x - xref)^2 dV = w_q |J|_q (x_k E_{kq} - xref(x_q))^2
        auto EM = EigenMap(E);
        x = EM.transpose() * numeric.segment(elNo * tensor::b::Shape[0], tensor::b::Shape[0]);
        double localError = 0;
        for (std::size_t q = 0; q < rule.size(); ++q) {
            double e = reference(&coords(0, q)) - x(q);
            localError += rule.weights()[q] * absDetJ(q) * e * e;
        }
        error += localError;
    }

    return sqrt(error);
}


template <std::size_t D, typename DFunc>
double ComputeError_H1semi(Curvilinear<D>& cl, Eigen::VectorXd const& numeric, DFunc grad_reference)
{
    auto rule = simplexQuadratureRule<D>(20);

    auto E = dubinerBasisAt(PolynomialDegree, rule.points());
    auto E_xi = dubinerBasisGradientAt(PolynomialDegree, rule.points()); // tensor[basis][deriv][point]
    auto geoE = cl.evaluateBasisAt(rule.points());
    auto geoD_xi = cl.evaluateGradientAt(rule.points());
    auto J = Managed(cl.jacobianResultInfo(rule.size()));
    auto absDetJ = Managed(cl.detJResultInfo(rule.size()));
    auto coords = Managed(cl.mapResultInfo(rule.size()));

    Eigen::VectorXd x(rule.size());

    double error = 0.0;
    for (std::size_t elNo = 0; elNo < cl.numElements(); ++elNo) {
        cl.jacobian(elNo, geoD_xi, J);
        cl.absDetJ(elNo, J, absDetJ);
        cl.map(elNo, geoE, coords);
        // int |u,x - uref,x|^2 dV + int |u,y - uref,y|^2 dV = w_q |J|_q (u_k,x E_{kq} - uref,x(x_q))^2 + w_q |J|_q (u_k,y E_{kq} - uref,y(x_q))^2
        auto EM = EigenMap(E);
        x = EM.transpose() * numeric.segment(elNo * tensor::b::Shape[0], tensor::b::Shape[0]);
     
	auto x_el = numeric.segment(elNo * tensor::b::Shape[0], tensor::b::Shape[0]);
	auto nbasis = tensor::b::Shape[0];
	std::cout << "Nb " << tensor::b::Shape[0] << std::endl;
	for (int i=0; i<nbasis; i++) {
		std::cout << "e " << elNo << " b_[" << i << "] = " << x_el[i] << std::endl;
	}

	//auto jq = J()[0];
	//std::cout << J(0)  << std::endl; 

	auto E_x = E_xi.subtensor(slice{}, 0, slice{});
	auto E_y = E_xi.subtensor(slice{}, 1, slice{});

	double localError = 0;
        for (std::size_t q = 0; q < rule.size(); ++q) {
            double gradXref[2], e;
	     
	    grad_reference(&coords(0, q),gradXref);
	    e = (gradXref[0] - 0) * (gradXref[0] - 0) + (gradXref[1] - 0) * (gradXref[1] - 0);

	    std::cout << "u,x " << gradXref[0] << " u,y "<< gradXref[1] << std::endl;
            localError += rule.weights()[q] * absDetJ(q) * e;
        }
        error += localError;
    }

    return sqrt(error);
}

} // namespace tndm

using tndm::Curvilinear;
using tndm::GenMesh;
using tndm::Poisson;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    argparse::ArgumentParser program("poisson");
    program.add_argument("-o").help("Output file name");
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
        // return v;
        // return {2.0 * v[0] - 1.0, 2.0 * v[1] - 1.0};
    };

    Curvilinear<2u> cl(*mesh, transform, PolynomialDegree);
    // Curvilinear<2u> cl(*mesh, transform);

    tndm::Stopwatch sw;

    // auto phi = [](double x) { return std::fabs(x) < 1.0 ? exp(-1.0 / (1.0 - x * x)) : 0.0; };
    // auto d2phidx2 = [&phi](double x) {
    // return std::fabs(x) < 1.0
    //? 2.0 * phi(x) * (-1.0 + 3 * std::pow(x, 4.0)) / std::pow(1.0 - x * x, 4.0)
    //: 0.0;
    //};

    sw.start();
    Poisson poisson(*mesh, cl, PolynomialDegree, MinQuadOrder());
    std::cout << "Constructed Poisson after " << sw.split() << std::endl;
    auto A = poisson.assemble();
    // auto b = poisson.rhs(
    //[&phi, &d2phidx2](std::array<double, 2> const& x) {
    // return -(d2phidx2(x[0]) * phi(x[1]) + phi(x[0]) * d2phidx2(x[1]));
    //},
    //[](std::array<double, 2> const&) {
    // return 0.0; });
    // auto b = poisson.rhs([](std::array<double, 2> const& x) { return 0.0; },
    //[](std::array<double, 2> const& x) { return x[0] + x[1]; });
    // auto b = poisson.rhs(
    //[](std::array<double, 2> const& x) {
    // return 2.0 * M_PI * M_PI * cos(M_PI * x[0]) * cos(M_PI * x[1]);
    //},
    //[](std::array<double, 2> const& x) { return cos(M_PI * x[0]) * cos(M_PI * x[1]); });
    // auto b = poisson.rhs(
    //[](std::array<double, 2> const& x) { return -12 * (x[0] * x[0] + x[1] * x[1]); },
    //[](std::array<double, 2> const& x) { return pow(x[0], 4.0) + pow(x[1], 4.0); });
    auto b = poisson.rhs(
        [](std::array<double, 2> const& x) {
            return (1.0 - 4.0 * x[1] * x[1]) * exp(-x[0] - x[1] * x[1]);
        },
        [](std::array<double, 2> const& x) { return exp(-x[0] - x[1] * x[1]); });

    // std::cout << A << std::endl;

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

    // double error = L2error(cl, x, [&phi](double x[]) { return phi(x[0]) * phi(x[1]); });
    // double error = L2error(cl, x, [](double x[]) { return x[0] + x[1]; });
    // double error = L2error(cl, x, [](double x[]) { return cos(M_PI * x[0]) * cos(M_PI * x[1]);
    // });
    // double error = L2error(cl, x, [](double x[]) { return pow(x[0], 4.0) + pow(x[1], 4.0); });
    double errorL2 =
        L2error(cl, x, [](double coords[]) { return exp(-coords[0] - coords[1] * coords[1]); });

    std::cout << "error(L2) " << errorL2 << std::endl;

    double errorH1 = ComputeError_H1semi(cl, x, [](double x[],double gu[]) { gu[0] = -exp(-x[0] - x[1]*x[1]); gu[1] = -2.0*x[1]*exp(-x[0] - x[1]*x[1]); });
    std::cout << "error(H1_s) " << errorH1 << std::endl;

    if (auto fileName = program.present("-o")) {
        std::vector<double> data(3 * mesh->numElements(), 0.0);
        auto outPoints = std::vector<std::array<double, 2u>>{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
        auto E = tndm::dubinerBasisAt(PolynomialDegree, outPoints);
        auto EM = EigenMap(E);
        for (std::size_t elNo = 0; elNo < mesh->numElements(); ++elNo) {
            Eigen::VectorXd out = EM.transpose() * x.segment(elNo * tndm::tensor::b::Shape[0],
                                                             tndm::tensor::b::Shape[0]);
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
        auto [cells, vertices] = duplicatedDofsMesh<unsigned int, double, 2u, 3u>(*mesh, transform);
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
