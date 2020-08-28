#include "SEAS.h"
#include "Elasticity.h"
#include "Fault.h"

#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "geometry/Vector.h"
#include "io/VTUAdapter.h"
#include "io/VTUWriter.h"
#include "util/Stopwatch.h"

#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscts.h>
#include <petscvec.h>

#include <iostream>
#include <memory>

namespace tndm {

struct SEASContext {
    std::unique_ptr<Elasticity> elasticity;
    std::unique_ptr<Fault> fault;
    std::unique_ptr<LuaScenario> scenario;
    Vec u;
    Vec b;
    KSP ksp;
    double timeAssembly = 0.0;
    double timeSolve = 0.0;
    double timeFault = 0.0;
    std::optional<std::string> output;
    std::unique_ptr<Curvilinear<DomainDimension>> cl;
};

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec x, Vec F, void* ctx) {
    SEASContext* user = reinterpret_cast<SEASContext*>(ctx);

    auto constant_integral = [t, user](std::array<double, DomainDimension> const& x) {
        return t * user->scenario->boundary()(x);
    };

    Stopwatch sw;
    sw.start();
    CHKERRQ(user->elasticity->rhs(
        user->b, user->elasticity->makeVolumeFunctional(user->scenario->force()),
        user->elasticity->makeFacetFunctional(constant_integral), user->fault->slip(x)));
    user->timeAssembly += sw.stop();

    sw.start();
    CHKERRQ(KSPSolve(user->ksp, user->b, user->u));
    user->timeSolve += sw.stop();

    sw.start();
    user->fault->rhs(*user->elasticity, user->u, x, F);
    user->timeFault += sw.stop();

    return 0;
}

PetscErrorCode monitor(TS ts, PetscInt step, PetscReal time, Vec X, void* ctx) {
    SEASContext* user = reinterpret_cast<SEASContext*>(ctx);

    unsigned step_interval = 1;
    if (user->output && step % step_interval == 0) {
        std::cout << "Output at time " << time << std::endl;

        auto fault = user->fault->finiteElementFunction(X);
        VTUWriter<DomainDimension - 1u> fWriter(PolynomialDegree, true, PETSC_COMM_WORLD);
        fWriter.addFieldData("time", &time, 1);
        auto fAdapter = CurvilinearBoundaryVTUAdapter(*user->cl, user->fault->elNos(),
                                                      user->fault->localFaceNos());
        auto fPiece = fWriter.addPiece(fAdapter);
        fPiece.addPointData("x", fault);
        std::stringstream fss;
        fss << *user->output << "-fault_" << (step / step_interval);
        fWriter.write(fss.str());

        auto numeric = user->elasticity->finiteElementFunction(user->u);
        VTUWriter<DomainDimension> writer(PolynomialDegree, true, PETSC_COMM_WORLD);
        writer.addFieldData("time", &time, 1);
        auto adapter = CurvilinearVTUAdapter(*user->cl, user->elasticity->numLocalElements());
        auto piece = writer.addPiece(adapter);
        piece.addPointData("u", numeric);
        std::stringstream ss;
        ss << *user->output << "_" << (step / step_interval);
        writer.write(ss.str());
    }
    return 0;
}

PetscErrorCode solveSEASProblem(Config const& cfg) {
    // Scenario

    Mat A;
    Vec psi;
    TS ts;
    SEASContext ctx;

    ctx.scenario = std::make_unique<LuaScenario>(cfg.problem);
    ctx.output = cfg.output;

    int rank, procs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &procs);

    auto meshGen = cfg.generate_mesh.create(cfg.resolution, PETSC_COMM_WORLD);
    auto globalMesh = meshGen.uniformMesh();
    globalMesh->repartition();
    auto mesh = globalMesh->getLocalMesh(1);

    ctx.cl = std::make_unique<Curvilinear<DomainDimension>>(*mesh, ctx.scenario->transform(),
                                                            PolynomialDegree);

    ctx.elasticity = std::make_unique<Elasticity>(
        *mesh, *ctx.cl, std::make_unique<tndm::ModalRefElement<DomainDimension>>(PolynomialDegree),
        MinQuadOrder(), PETSC_COMM_WORLD, ctx.scenario->lam(), ctx.scenario->mu());
    ctx.fault = std::make_unique<Fault>(*mesh, *ctx.cl, ctx.elasticity->facetRule().points(),
                                        PETSC_COMM_WORLD);

    // Assemble
    {
        auto interface = ctx.elasticity->interfacePetsc();
        CHKERRQ(interface.createA(&A));
        CHKERRQ(interface.createb(&ctx.b));
    }
    CHKERRQ(ctx.elasticity->assemble(A));
    CHKERRQ(ctx.elasticity->rhs(ctx.b, ctx.scenario->force(), ctx.scenario->boundary(),
                                ctx.scenario->slip()));
    CHKERRQ(VecDuplicate(ctx.b, &ctx.u));

    CHKERRQ(ctx.fault->createState(&psi));
    CHKERRQ(ctx.fault->initial(psi));

    // Setup KSP
    CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ctx.ksp));
    CHKERRQ(KSPSetType(ctx.ksp, KSPCG));
    CHKERRQ(KSPSetOperators(ctx.ksp, A, A));
    CHKERRQ(KSPSetTolerances(ctx.ksp, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    CHKERRQ(KSPSetFromOptions(ctx.ksp));

    // Setup TS
    CHKERRQ(TSCreate(PETSC_COMM_WORLD, &ts));
    CHKERRQ(TSSetProblemType(ts, TS_NONLINEAR));
    CHKERRQ(TSSetSolution(ts, psi));
    CHKERRQ(TSSetRHSFunction(ts, nullptr, RHSFunction, &ctx));
    CHKERRQ(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    CHKERRQ(TSMonitorSet(ts, monitor, &ctx, nullptr));
    CHKERRQ(TSSetFromOptions(ts));
    CHKERRQ(TSSetUp(ts));

    // Solve
    // double sync_interval = cfg.output_interval ? *cfg.output_interval : cfg.final_time;
    double sync_interval = cfg.final_time;
    double time = 0.0;
    std::size_t step = 0;
    while (time < cfg.final_time) {
        double upcoming_time = std::min(time + sync_interval, cfg.final_time);
        CHKERRQ(TSSetMaxTime(ts, upcoming_time));
        CHKERRQ(TSSolve(ts, psi));
        time = upcoming_time;
        std::cout << "Times: " << ctx.timeAssembly << ", " << ctx.timeSolve << ", " << ctx.timeFault
                  << std::endl;
    }

    // Clean up
    CHKERRQ(KSPDestroy(&ctx.ksp));
    CHKERRQ(TSDestroy(&ts));
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(VecDestroy(&ctx.b));
    CHKERRQ(VecDestroy(&ctx.u));
    CHKERRQ(VecDestroy(&psi));

    return 0;
}

} // namespace tndm
