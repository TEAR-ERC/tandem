#include "Static.h"
#include "Elasticity.h"

#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "io/VTUWriter.h"
#include "util/Stopwatch.h"

#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>

#include <iostream>

namespace tndm {

PetscErrorCode solveStaticProblem(Config const& cfg) {
    auto scenario = std::make_unique<LuaScenario>(cfg.problem);

    int rank, procs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &procs);

    auto meshGen = cfg.generate_mesh.create(cfg.resolution, PETSC_COMM_WORLD);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();
    auto mesh = globalMesh->getLocalMesh(1);

    Curvilinear<DomainDimension> cl(*mesh, scenario->transform(), PolynomialDegree);

    Stopwatch sw;
    sw.start();
    Elasticity elasticity(
        *mesh, cl, std::make_unique<tndm::ModalRefElement<DomainDimension>>(PolynomialDegree),
        MinQuadOrder(), PETSC_COMM_WORLD, scenario->lam(), scenario->mu());
    std::cout << "Constructed Poisson after " << sw.split() << std::endl;

    Mat A;
    Vec b, x, y;
    KSP ksp;

    {
        auto interface = elasticity.interfacePetsc();
        CHKERRQ(interface.createA(&A));
        CHKERRQ(interface.createb(&b));
    }
    CHKERRQ(elasticity.assemble(A));
    CHKERRQ(elasticity.rhs(b, scenario->force(), scenario->boundary(), scenario->slip()));
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
    auto solution = scenario->solution();
    if (solution) {
        double error =
            tndm::Error<DomainDimension>::L2(cl, numeric, *solution, 0, PETSC_COMM_WORLD);
        if (rank == 0) {
            std::cout << "L2 error: " << error << std::endl;
        }
    }

    if (cfg.output) {
        VTUWriter<DomainDimension> writer(PolynomialDegree, true, PETSC_COMM_WORLD);
        auto piece = writer.addPiece(cl, elasticity.numLocalElements());
        piece.addPointData("u", numeric);
        piece.addPointData("lam", elasticity.discreteLambda());
        piece.addPointData("mu", elasticity.discreteMu());
        writer.write(*cfg.output);
    }

    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y));
    return 0;
}

} // namespace tndm
