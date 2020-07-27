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
    case "embedded"_fnv1a: {
        auto scenario = std::make_unique<MyScenario>(
            biunit, [](std::array<double, 2> const& x) { return 0.0; },
            [](std::array<double, 2> const& x) { return x[0] < 0.5 ? 3.0 : 2.0; },
            [](Vector<double> const& x) -> std::array<double, 1> {
                return {x(0) < 0.5 ? 3.0 : 2.0};
            });
        auto xbc = [](std::size_t plane, std::array<std::size_t, 1> const&) {
            if (plane == 1) {
                return BC::Fault;
            }
            return BC::Dirichlet;
        };
        auto ybc = [](std::size_t plane, std::array<std::size_t, 1> const&) {
            if (plane == 1) {
                return BC::Dirichlet;
            }
            return BC::Natural;
        };
        scenario->setPointsAndBCs({{{0.0, 0.75, 1.0}, {0.0, 1.0}}}, {{xbc, ybc}});
        return scenario;
    }
    case "embedded_half"_fnv1a: {
        auto smoothstep = [](double x) {
            return 6.0 * std::pow(x, 5.0) - 15.0 * std::pow(x, 4.0) + 10.0 * std::pow(x, 3.0);
        };
        auto dsmoothstep_dx2 = [](double x) {
            return 120.0 * std::pow(x, 3.0) - 180.0 * std::pow(x, 2.0) + 60.0 * x;
        };
        auto scenario = std::make_unique<MyScenario>(
            biunit,
            [dsmoothstep_dx2](std::array<double, 2> const& x) {
                auto sign = x[0] > 0.5 ? -1.0 : 1.0;
                return x[1] > 0.0 ? -sign * dsmoothstep_dx2(x[1]) : 0.0;
            },
            [smoothstep](std::array<double, 2> const& x) {
                auto sign = x[0] > 0.5 ? -1.0 : 1.0;
                return x[1] > 0.0 ? sign * smoothstep(x[1]) : 0.0;
            },
            [smoothstep](Vector<double> const& x) -> std::array<double, 1> {
                auto sign = x(0) > 0.5 ? -1.0 : 1.0;
                return {x(1) > 0.0 ? sign * smoothstep(x(1)) : 0.0};
            });
        auto xbc = [](std::size_t plane, std::array<std::size_t, 1> const& regions) {
            if (plane == 1) {
                if (regions[0] == 1) {
                    return BC::Fault;
                }
                return BC::None;
            }
            return BC::Natural;
        };
        auto ybc = [](std::size_t plane, std::array<std::size_t, 1> const&) {
            if (plane == 1) {
                return BC::None;
            }
            return BC::Dirichlet;
        };
        scenario->setPointsAndBCs({{{0.0, 0.75, 1.0}, {0.0, 0.5, 1.0}}}, {{xbc, ybc}});
        return scenario;
    }
    default:
        return nullptr;
    }
    return nullptr;
}

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

    argparse::ArgumentParser program("poisson");
    program.add_argument("--petsc").help("PETSc options, must be passed last!");
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

    int rank, procs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &procs);

    auto n = program.get<unsigned long>("n");
    auto globalMesh = scenario->getGlobalMesh(n, PETSC_COMM_WORLD);
    auto mesh = globalMesh->getLocalMesh(1);

    Curvilinear<DomainDimension> cl(*mesh, scenario->transform(), PolynomialDegree);

    tndm::Stopwatch sw;

    sw.start();
    Poisson poisson(*mesh, cl, std::make_unique<tndm::ModalRefElement<2ul>>(PolynomialDegree),
                    MinQuadOrder(), PETSC_COMM_WORLD, scenario->K());
    std::cout << "Constructed Poisson after " << sw.split() << std::endl;

    Mat A;
    Vec b, x, y;
    KSP ksp;

    {
        auto interface = poisson.interfacePetsc();
        CHKERRQ(interface.createA(&A));
        CHKERRQ(interface.createb(&b));
    }

    CHKERRQ(poisson.assemble(A));
    CHKERRQ(poisson.rhs(b, scenario->force(), scenario->dirichlet()));
    std::cout << "Assembled after " << sw.split() << std::endl;

    CHKERRQ(VecDuplicate(b, &x));
    CHKERRQ(VecDuplicate(b, &y));
    CHKERRQ(VecSet(x, 1.0));
    CHKERRQ(MatMult(A, x, y));
    PetscReal l2norm;
    CHKERRQ(VecNorm(y, NORM_2, &l2norm));
    std::cout << "A*1 norm: " << l2norm << std::endl;

    CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ksp));
    CHKERRQ(KSPSetType(ksp, KSPCG));
    CHKERRQ(KSPSetOperators(ksp, A, A));
    CHKERRQ(KSPSetTolerances(ksp, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    CHKERRQ(KSPSetFromOptions(ksp));

    /* If you want to use the BAIJ operator on the finest level, forcefully insert it */
    /*
    {
        KSP      smoother;
        PC       pc;
        PetscInt nlevels;

        ierr = KSPSetUp(ksp);CHKERRQ(ierr);
        ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
        ierr = PCMGGetLevels(pc,&nlevels);CHKERRQ(ierr);
        ierr = PCMGGetSmoother(pc,nlevels-1,&smoother);CHKERRQ(ierr);
        ierr = KSPSetOperators(smoother, Aaij, A);CHKERRQ(ierr);
        ierr = KSPSetUp(smoother);CHKERRQ(ierr);
    }
    */

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

    auto numeric = poisson.finiteElementFunction(x);
    double error =
        tndm::Error<DomainDimension>::L2(cl, numeric, *scenario->reference(), 0, PETSC_COMM_WORLD);

    if (rank == 0) {
        std::cout << "L2 error: " << error << std::endl;
    }

    if (auto fileName = program.present("-o")) {
        VTUWriter<2u> writer(PolynomialDegree, true, PETSC_COMM_WORLD);
        auto piece = writer.addPiece(cl, poisson.numLocalElements());
        piece.addPointData("u", numeric);
        piece.addPointData("K", poisson.discreteK());
        writer.write(*fileName);
    }

    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y));

    ierr = PetscFinalize();

    return ierr;
}
