#include "SEAS.h"
#include "Fault.h"
#include "common/InterfacePetsc.h"
#include "common/MeshConfig.h"
#include "config.h"
#include "poisson/Poisson.h"
#include "tandem/Config.h"
#include "tandem/Scenario.h"
#include "tensor/Managed.h"

#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "io/GMSHParser.h"
#include "io/GlobalSimplexMeshBuilder.h"
#include "io/VTUAdapter.h"
#include "io/VTUWriter.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "quadrules/SimplexQuadratureRule.h"
#include "util/Range.h"
#include "util/Stopwatch.h"

#include <mpi.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscts.h>
#include <petscvec.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tndm {

struct SEASContext {
    std::unique_ptr<Poisson> poisson;
    std::unique_ptr<Fault> fault;
    std::unique_ptr<LuaScenario> scenario;
    Mat A;
    Vec u;
    Vec b;
    KSP ksp;
    double timeAssembly = 0.0;
    double timeSolve = 0.0;
    double timeFault = 0.0;
    std::optional<std::string> output;
    std::unique_ptr<Curvilinear<DomainDimension>> cl;
    double lastOutputTime = std::numeric_limits<double>::lowest();
    std::size_t outputNo = 0;
};

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec x, Vec F, void* ctx) {
    SEASContext* user = reinterpret_cast<SEASContext*>(ctx);

    auto constant_integral = [t, user](std::array<double, DomainDimension> const& x) {
        return t * user->scenario->boundary()(x);
    };

    Stopwatch sw2;
    sw2.start();
    Stopwatch sw;
    sw.start();
    VecZeroEntries(user->b);

    CHKERRQ(user->poisson->rhs(
        user->b, user->poisson->makeVolumeFunctional(user->scenario->force()),
        user->poisson->makeFacetFunctional(constant_integral), user->fault->slip(x)));
    user->timeAssembly += sw.stop();

    sw.start();
    CHKERRQ(KSPSolve(user->ksp, user->b, user->u));
    user->timeSolve += sw.stop();

    sw.start();
    user->fault->rhs(*user->poisson, user->u, x, F);
    user->timeFault += sw.stop();
    std::cout << sw2.stop() << std::endl;

    return 0;
}

PetscErrorCode monitor(TS ts, PetscInt step, PetscReal time, Vec X, void* ctx) {
    SEASContext* user = reinterpret_cast<SEASContext*>(ctx);

    constexpr double V0 = 1e-9;
    constexpr double V1 = 1;
    constexpr double tmin = 0.01;
    constexpr double tmax = 365 * 24 * 3600;
    double falloff = log(tmin / tmax) * (V1 - V0);
    auto VMax = user->fault->getVMax();
    VMax = std::min(1.0, std::max(1e-9, VMax)); // Clamp to [1e-9, 1]
    double outputInterval = tmax * exp(falloff * VMax);
    std::cout << "Output interval: " << outputInterval << std::endl;
    if (user->output && time - user->lastOutputTime >= outputInterval) {
        std::cout << "Output at time " << time << std::endl;

        auto fault = user->fault->finiteElementFunction(X);
        VTUWriter<DomainDimension - 1u> fWriter(PolynomialDegree, true, PETSC_COMM_WORLD);
        fWriter.addFieldData("time", &time, 1);
        auto fAdapter = CurvilinearBoundaryVTUAdapter(*user->cl, user->fault->elNos(),
                                                      user->fault->localFaceNos());
        auto fPiece = fWriter.addPiece(fAdapter);
        fPiece.addPointData("x", fault);
        std::stringstream fss;
        fss << *user->output << "-fault_" << user->outputNo;
        fWriter.write(fss.str());

        auto numeric = user->poisson->finiteElementFunction(user->u);
        VTUWriter<DomainDimension> writer(PolynomialDegree, true, PETSC_COMM_WORLD);
        writer.addFieldData("time", &time, 1);
        auto adapter = CurvilinearVTUAdapter(*user->cl, user->poisson->numLocalElements());
        auto piece = writer.addPiece(adapter);
        piece.addPointData("u", numeric);
        std::stringstream ss;
        ss << *user->output << "_" << user->outputNo;
        writer.write(ss.str());

        user->lastOutputTime = time;
        ++user->outputNo;
    }
    return 0;
}

PetscErrorCode solveSEASProblem(Config const& cfg) {
    // Scenario

    Vec psi;
    TS ts;
    SEASContext ctx;

    ctx.scenario = std::make_unique<LuaScenario>(cfg.problem);
    ctx.output = cfg.output;

    int rank, procs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &procs);

    std::unique_ptr<GlobalSimplexMesh<DomainDimension>> globalMesh;
    if (cfg.mesh_file) {
        bool ok = false;
        GlobalSimplexMeshBuilder<DomainDimension> builder;
        if (rank == 0) {
            GMSHParser parser(&builder);
            ok = parser.parseFile(*cfg.mesh_file);
            if (!ok) {
                std::cerr << *cfg.mesh_file << std::endl << parser.getErrorMessage();
            }
        }
        MPI_Bcast(&ok, 1, MPI_CXX_BOOL, 0, PETSC_COMM_WORLD);
        if (ok) {
            globalMesh = builder.create(PETSC_COMM_WORLD);
        }
        if (procs > 1) {
            // ensure initial element distribution for metis
            globalMesh->repartitionByHash();
        }
    } else if (cfg.generate_mesh && cfg.resolution) {
        auto meshGen = cfg.generate_mesh->create(*cfg.resolution, PETSC_COMM_WORLD);
        globalMesh = meshGen.uniformMesh();
    }
    if (!globalMesh) {
        std::cerr
            << "You must either provide a valid mesh file or provide the mesh generation config "
               "(including the resolution parameter)."
            << std::endl;
        PetscFinalize();
        return -1;
    }
    globalMesh->repartition();
    auto mesh = globalMesh->getLocalMesh(1);

    ctx.cl = std::make_unique<Curvilinear<DomainDimension>>(*mesh, ctx.scenario->transform(),
                                                            PolynomialDegree);

    ctx.poisson = std::make_unique<Poisson>(
        *mesh, *ctx.cl, std::make_unique<tndm::ModalRefElement<DomainDimension>>(PolynomialDegree),
        MinQuadOrder(), PETSC_COMM_WORLD, ctx.scenario->mu());
    ctx.fault = std::make_unique<Fault>(*mesh, *ctx.cl, ctx.poisson->facetRule().points(),
                                        PETSC_COMM_WORLD);

    // Assemble
    {
        auto interface = ctx.poisson->interfacePetsc();
        CHKERRQ(interface.createA(&ctx.A));
        CHKERRQ(interface.createb(&ctx.b));
    }
    CHKERRQ(ctx.poisson->assemble(ctx.A));
    CHKERRQ(ctx.poisson->rhs(ctx.b, ctx.scenario->force(), ctx.scenario->boundary(),
                             ctx.scenario->slip()));
    CHKERRQ(VecDuplicate(ctx.b, &ctx.u));

    CHKERRQ(ctx.fault->createState(&psi));
    CHKERRQ(ctx.fault->initial(psi));

    // Setup KSP
    CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ctx.ksp));
    CHKERRQ(KSPSetType(ctx.ksp, KSPCG));
    CHKERRQ(KSPSetOperators(ctx.ksp, ctx.A, ctx.A));
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
    CHKERRQ(MatDestroy(&ctx.A));
    CHKERRQ(VecDestroy(&ctx.b));
    CHKERRQ(VecDestroy(&ctx.u));
    CHKERRQ(VecDestroy(&psi));

    return 0;
}

} // namespace tndm
