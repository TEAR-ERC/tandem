#include "config.h"
#include "tandem/Elasticity.h"

#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "io/VTUWriter.h"
#include "mesh/GenMesh.h"
#include "mesh/MeshData.h"
#include "tensor/Tensor.h"
#include "util/Stopwatch.h"

#include <argparse.hpp>
#include <mpi.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <iostream>
#include <memory>
#include <tuple>

using tndm::BC;
using tndm::Curvilinear;
using tndm::Elasticity;
using tndm::GenMesh;
using tndm::LambdaSolution;
using tndm::Vector;
using tndm::VertexData;
using tndm::VTUWriter;

int main(int argc, char** argv) {
    int pArgc = 0;
    char** pArgv = nullptr;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--petsc") == 0) {
            pArgc = argc - i;
            pArgv = argv + i;
            argc = i;
            break;
        }
    }
    PetscErrorCode ierr;
    CHKERRQ(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));

    argparse::ArgumentParser program("tandem");
    program.add_argument("--petsc").help("PETSc options, must be passed last!");
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

    int rank, procs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &procs);

    auto n = program.get<unsigned long>("n");
    std::array<uint64_t, DomainDimension> size = {4 * n, n};
    std::array<std::pair<BC, BC>, DomainDimension> BCs = {{
        //{BC::Dirichlet, BC::Natural},
        //{BC::Natural, BC::Natural},
        {BC::Dirichlet, BC::Dirichlet},
        {BC::Dirichlet, BC::Dirichlet},
    }};
    auto meshGen = GenMesh(size, BCs, PETSC_COMM_WORLD);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();
    auto mesh = globalMesh->getLocalMesh(1);

    Curvilinear<DomainDimension> cl(*mesh,
                                    [](std::array<double, 2> const& v) -> std::array<double, 2> {
                                        return {4.0 * v[0], v[1]};
                                    });

    tndm::Stopwatch sw;
    sw.start();
    Elasticity elasticity(
        *mesh, cl, std::make_unique<tndm::ModalRefElement<DomainDimension>>(PolynomialDegree),
        MinQuadOrder(), PETSC_COMM_WORLD, [](auto) { return 1.0; }, [](auto) { return 1.0; });
    std::cout << "Constructed Poisson after " << sw.split() << std::endl;

    Mat A;
    Vec b, x, y;
    KSP ksp;

    auto forceFun = [](auto const& x) -> std::array<double, DomainDimension> {
        return {M_PI * M_PI * 4.0 * cos(M_PI * x[0]) * cos(M_PI * x[1]),
                -M_PI * M_PI * 2.0 * sin(M_PI * x[0]) * sin(M_PI * x[1])};
    };
    auto dirichletFun = [](auto const& x) -> std::array<double, DomainDimension> {
        return {cos(M_PI * x[0]) * cos(M_PI * x[1]), 0.0};
    };

    {
        auto interface = elasticity.interfacePetsc();
        CHKERRQ(interface.createA(&A));
        CHKERRQ(interface.createb(&b));
    }
    CHKERRQ(elasticity.assemble(A));
    CHKERRQ(elasticity.rhs(b, forceFun, dirichletFun));
    std::cout << "Assembled after " << sw.split() << std::endl;

    CHKERRQ(VecDuplicate(b, &x));
    CHKERRQ(VecDuplicate(b, &y));
    CHKERRQ(VecSet(x, 1.0));
    CHKERRQ(MatMult(A, x, y));
    sw.start();
    for (int i = 0; i < 100; ++i) {
        CHKERRQ(MatMult(A, x, y));
    }
    double timeA100 = sw.split();
    std::cout << "A time: " << timeA100 / 100 << std::endl;
    PetscReal l2norm;
    CHKERRQ(VecNorm(y, NORM_2, &l2norm));
    std::cout << "A*1 norm: " << l2norm << std::endl;

    CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ksp));
    CHKERRQ(KSPSetType(ksp, KSPCG));
    CHKERRQ(KSPSetOperators(ksp, A, A));
    CHKERRQ(KSPSetTolerances(ksp, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    CHKERRQ(KSPSetFromOptions(ksp));

    CHKERRQ(KSPSolve(ksp, b, x));
    std::cout << "Solved after " << sw.split() << std::endl;
    PetscReal rnorm;
    PetscInt its;
    CHKERRQ(KSPGetResidualNorm(ksp, &rnorm));
    CHKERRQ(KSPGetIterationNumber(ksp, &its));
    if (rank == 0) {
        std::cout << "Residual norm: " << rnorm << std::endl;
        std::cout << "Iterations: " << its << std::endl;
    }

    CHKERRQ(KSPDestroy(&ksp));
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(VecDestroy(&b));

    auto numeric = elasticity.finiteElementFunction(x);
    auto solution =
        LambdaSolution([](Vector<double> const& x) -> std::array<double, DomainDimension> {
            return {cos(M_PI * x(0)) * cos(M_PI * x(1)), 0.0};
        });
    double error = tndm::Error<DomainDimension>::L2(cl, numeric, solution, 0, PETSC_COMM_WORLD);

    if (rank == 0) {
        std::cout << "L2 error: " << error << std::endl;
    }

    if (auto fileName = program.present("-o")) {
        VTUWriter<2u> writer(PolynomialDegree, true, PETSC_COMM_WORLD);
        auto piece = writer.addPiece(cl, elasticity.numLocalElements());
        piece.addPointData("u", numeric);
        piece.addPointData("lam", elasticity.discreteLambda());
        piece.addPointData("mu", elasticity.discreteMu());
        writer.write(*fileName);
    }

    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y));

    ierr = PetscFinalize();

    return ierr;
}
