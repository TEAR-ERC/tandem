#include "config.h"

#include "kernels/kernel.h"
#include "kernels/tensor.h"

#include "form/DG.h"
#include "geometry/Curvilinear.h"
#include "mesh/GenMesh.h"

#include <argparse.hpp>
#include <mpi.h>

#include <cassert>
#include <iostream>

namespace tndm {

class Poisson : DG<DomainDimension> {
public:
    using DG<DomainDimension>::DG;

    void assemble() {
        double Dx[tensor::Dx::size()];
        double A[tensor::A::size()];

        assert(volRule.size() == tensor::absDetJ::Shape[0]);
        assert(Dxi.shape(0) == tensor::Dxi::Shape[0]);
        assert(Dxi.shape(1) == tensor::Dxi::Shape[1]);
        assert(Dxi.shape(2) == tensor::Dxi::Shape[2]);

        for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
            kernel::assembly krnl;
            krnl.A = A;
            krnl.Dx = Dx;
            krnl.Dxi = Dxi.data();
            krnl.G = vol[elNo].template get<JInvT>().data()->data();
            krnl.absDetJ = vol[elNo].template get<AbsDetJ>().data();
            krnl.w = volRule.weights().data();
            krnl.execute();
        }
    }
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

    Curvilinear cl(*mesh);

    Poisson poisson(*mesh, cl, PolynomialDegree, MinQuadOrder());
    poisson.assemble();

    MPI_Finalize();

    return 0;
}
