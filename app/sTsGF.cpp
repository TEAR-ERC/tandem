#include "common/Banner.h"
#include "common/CmdLine.h"
#include "common/ElasticityScenario.h"
#include "common/MGConfig.h"
#include "common/MeshConfig.h"
#include "common/PetscLinearSolver.h"
#include "common/PetscUtil.h"
#include "common/PoissonScenario.h"
#include "common/Type.h"
#include "config.h"
#include "form/DGCurvilinearCommon.h"
#include "mesh/LocalSimplexMesh.h"
#include "pc/register.h"

#include "form/DGOperator.h"
#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "io/GlobalSimplexMeshBuilder.h"
#include "io/MeshParser.h"
#include "io/VTUAdapter.h"
#include "io/VTUWriter.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "parallel/Affinity.h"
#include "tensor/Managed.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"
#include "util/Stopwatch.h"

#include <argparse.hpp>
#include <limits>
#include <mpi.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <stdexcept>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "quadrules/SimplexQuadratureRule.h"
#include "form/AdapterOperator.h"
#include "form/BoundaryMap.h"
#include "localoperator/Adapter.h"
#include "localoperator/RateAndStateBase.h"
#include "sTsGF_receivers.h"

using namespace tndm;

struct Config {
    std::optional<double> resolution;
    DGMethod method;
    LocalOpType type;
    std::string lib;
    std::string scenario;
    std::array<double, DomainDimension> ref_normal;
    bool matrix_free;
    bool test_matrix_free;
    MGStrategy mg_strategy;
    unsigned mg_coarse_level;
    int profile;
    std::optional<std::string> output;
    std::optional<std::string> mesh_file;
    std::optional<GenMeshConfig<DomainDimension>> generate_mesh;
    std::array<double, DomainDimension> up;
    long int receiver_surface_tag;
    std::size_t receiver_grid_n;
};

/* Sampling "rule" whose points are the nodes of the fault space. */
auto nodal_sampling_rule() -> SimplexQuadratureRule<DomainDimension - 1u> {
    auto space = RateAndStateBase::Space();
    auto const& nodes = space.refNodes();
    auto rule = SimplexQuadratureRule<DomainDimension - 1u>(nodes.size(), -1);
    rule.points() = nodes;
    // Only used for the mass matrix, which this adapter never applies.
    std::fill(rule.weights().begin(), rule.weights().end(), 1.0 / nodes.size());
    return rule;
}

/**
 * Strike, dip and the unit normal at every node of every local fault facet.
 *
 * A throwaway AdapterOperator built on the nodal sampling rule. Its prepare()
 * runs exactly the same Curvilinear::normal and Curvilinear::facetBasis calls
 * as the real one, just evaluated at the nodes.
 */
template <class LocalOperator> struct slip_traits;
template <> struct slip_traits<Elasticity> {
    static constexpr std::size_t NumSlipComponents = DomainDimension - 1u;
};
template <> struct slip_traits<Poisson> {
    static constexpr std::size_t NumSlipComponents = 1u;
};

/**
 * Put unit slip on every facet carrying gfTag, in the given fault-basis
 * direction, and assemble b for it.
 *
 * Same sequence as SeasQDOperator::solve.
 */
template <class LocalOperator>
void set_slip_and_rhs(long int gfTag, std::size_t direction,
                      std::vector<long int> const& faultNo2tag, std::size_t nbf_fault,
                      PetscVector& S, Scatter& scatter, SparseBlockVector<double>& ghost,
                      AdapterOperator<LocalOperator>& adapter, DGOperator<LocalOperator>& dgop,
                      PetscLinearSolver& solver) {
    /*
     * The fault space is nodal, so setting all nbf coefficients of one
     * direction to 1 is exactly unit slip over the patch. The block layout
     * matches the yateto tensor slip(nbf_fault, D-1), first index fastest.
     */
    S.set_zero();
    {
        auto handle = S.begin_access();
        for (std::size_t faultNo = 0; faultNo < faultNo2tag.size(); ++faultNo) {
            if (faultNo2tag[faultNo] != gfTag) {
                continue;
            }
            for (std::size_t l = 0; l < nbf_fault; ++l) {
                handle(l + direction * nbf_fault, faultNo) = 1.0;
            }
        }
        S.end_access(handle);
    }

    // Fault facets shared with another rank need the owner's slip.
    scatter.begin_scatter(S, ghost);
    scatter.wait_scatter();
    auto S_view = LocalGhostCompositeView(S, ghost);

    /*
     * slip_bc rotates the nodal slip into the fault basis and evaluates it at
     * the facet quadrature points. update_rhs zeroes b and calls dgop.rhs(b),
     * so the Lua force and Dirichlet terms come along in the same pass.
     */
    dgop.set_slip(adapter.slip_bc(S_view));
    solver.update_rhs(dgop);

    // S_view dies at the end of this scope, so drop the functional holding it.
    dgop.set_slip([](std::size_t, Matrix<double>&, bool) {
        throw std::logic_error("Slip boundary condition not set");
    });
}
template <class LocalOperator>
auto fault_angle_function(std::shared_ptr<Curvilinear<DomainDimension>> cl,
                          std::shared_ptr<LocalOperator> lop,
                          std::shared_ptr<DGOperatorTopo> topo,
                          std::shared_ptr<BoundaryMap> fault_map,
                          std::array<double, DomainDimension> const& up,
                          std::array<double, DomainDimension> const& ref_normal)
    -> FiniteElementFunction<DomainDimension - 1u> {
    static_assert(DomainDimension == 3u, "strike and dip are 3D only");

    auto cross = [](std::array<double, 3> const& a, std::array<double, 3> const& b) {
        auto c = std::array<double, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                                       a[0] * b[1] - a[1] * b[0]};
        auto l = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        return std::array<double, 3>{c[0] / l, c[1] / l, c[2] / l};
    };

    auto space = RateAndStateBase::Space();
    auto rule = nodal_sampling_rule();

    auto probe = AdapterOperator<LocalOperator>(
        lop, std::make_unique<Adapter<LocalOperator>>(cl, space, rule, up, ref_normal), topo,
        fault_map);
    auto const& a = probe.lop();

    auto nbf = space.numBasisFunctions();
    auto num = fault_map->local_size();

    auto f = FiniteElementFunction<DomainDimension - 1u>(
        space.clone(),
        std::vector<std::string>{"strike", "dip", "tag", "nx", "ny", "nz", "dip_dir_x",
                                 "dip_dir_y", "dip_dir_z", "strike_dir_x", "strike_dir_y",
                                 "strike_dir_z"},
        num);
    auto& v = f.values();

    for (std::size_t faultNo = 0; faultNo < num; ++faultNo) {
        auto tag = static_cast<double>(topo->info(fault_map->fctNo(faultNo)).facetTag);

        for (std::size_t l = 0; l < nbf; ++l) {
            auto const& B = a.fault_basis(faultNo, l);

            /*
             * Column 0 is the raw mesh normal. prepare() flips it against
             * ref_normal, builds the basis, then negates the whole basis again,
             * so the stored columns carry the uncorrected orientation. Redo the
             * correction and rebuild the frame the way facetBasis does.
             */
            std::array<double, 3> n{B[0], B[1], B[2]};
            if (n[0] * ref_normal[0] + n[1] * ref_normal[1] + n[2] * ref_normal[2] < 0.0) {
                n = {-n[0], -n[1], -n[2]};
            }

            auto s = cross(up, n); // strike direction
            auto d = cross(s, n);  // dip direction

            auto az = std::atan2(s[0], s[1]) * 180.0 / M_PI;
            if (az < 0.0) {
                az += 360.0;
            }
            if (az > 360.0 - 1e-9) {
                az = 0.0;
            }

            v(l, 0, faultNo) = az;
            v(l, 1, faultNo) = std::acos(std::clamp(std::fabs(n[2]), 0.0, 1.0)) * 180.0 / M_PI;
            v(l, 2, faultNo) = tag;
            v(l, 3, faultNo) = n[0];
            v(l, 4, faultNo) = n[1];
            v(l, 5, faultNo) = n[2];
            v(l, 6, faultNo) = d[0];
            v(l, 7, faultNo) = d[1];
            v(l, 8, faultNo) = d[2];
            v(l, 9, faultNo) = s[0];
            v(l, 10, faultNo) = s[1];
            v(l, 11, faultNo) = s[2];
        }
    }
    return f;
}
std::set<long int>
get_gf_tags(DGOperatorTopo const& topo) {
    /*
     * Find distinct BC::Fault facet tags on this MPI rank.
     * Each distinct physical facet tag represents one GF source.
     */
    std::set<long int> localGfTags;

    for (std::size_t fctNo = 0; fctNo < topo.numLocalFacets(); ++fctNo) {
        auto const& info = topo.info(fctNo);

        if (info.bc == BC::Fault) {
            localGfTags.insert(info.facetTag);
        }
    }

    /*
     * Gather tags from all MPI ranks.
     *
     * Every rank must know the complete GF source list because
     * KSPSolve is collective: all ranks must perform the same
     * sequence of source solves.
     */
    int mpiSize;
    MPI_Comm_size(topo.comm(), &mpiSize);

    std::vector<long int> localTags(
        localGfTags.begin(),
        localGfTags.end());

    int nLocal = static_cast<int>(localTags.size());

    std::vector<int> counts(mpiSize);

    MPI_Allgather(&nLocal, 1, MPI_INT,counts.data(),1,MPI_INT,topo.comm());

    std::vector<int> displs(mpiSize, 0);

    for (int rank = 1; rank < mpiSize; ++rank) {
        displs[rank] =
            displs[rank - 1] + counts[rank - 1];
    }

    int nGlobal =
        displs.back() + counts.back();

    std::vector<long int> allTags(nGlobal);

    MPI_Allgatherv(localTags.data(),nLocal,MPI_LONG,allTags.data(),counts.data(),displs.data(),MPI_LONG,topo.comm());
    /*
     * std::set removes duplicates and sorts the tags.
     */
    return std::set<long int>(
        allTags.begin(),
        allTags.end());
}

void write_fault_vtu(LocalSimplexMesh<DomainDimension> const& mesh,
                     std::shared_ptr<Curvilinear<DomainDimension>> cl,
                     std::vector<std::size_t> const& fctNos,
                     FiniteElementFunction<DomainDimension - 1u> const& f,
                     std::string const& filename) {
    auto adapter = CurvilinearBoundaryVTUAdapter(mesh, cl, fctNos);
    auto writer = VTUWriter<DomainDimension - 1u>(PolynomialDegree, true, PETSC_COMM_WORLD);
    auto& piece = writer.addPiece(adapter);
    piece.addPointData(f);
    writer.write(filename);
}

template <class Scenario>
void static_problem(LocalSimplexMesh<DomainDimension> const& mesh, Scenario const& scenario,
                    Config const& cfg) {
    tndm::Stopwatch sw;
    double time;

    PetscLogStage solve;
    PetscLogStageRegister("solve", &solve);

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    auto cl = std::make_shared<Curvilinear<DomainDimension>>(mesh, scenario.transform(),
                                                             PolynomialDegree);

    auto lop = scenario.make_local_operator(cl, cfg.method);
    auto topo = std::make_shared<DGOperatorTopo>(mesh, PETSC_COMM_WORLD);
    auto gfTags = get_gf_tags(*topo);
    auto dgop = DGOperator(topo, lop);   

    const auto reduce_number = [&topo](std::size_t number) {
        std::size_t number_global;
        MPI_Reduce(&number, &number_global, 1, mpi_type_t<std::size_t>(), MPI_SUM, 0, topo->comm());
        return number_global;
    };

    if (cfg.test_matrix_free) {
        auto A = std::make_unique<PetscDGShell>(dgop);
        Vec x, y;
        CHKERRTHROW(MatCreateVecs(A->mat(), &x, &y));
        PetscRandom rctx;
        CHKERRTHROW(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
        CHKERRTHROW(VecSetRandom(x, rctx));
        CHKERRTHROW(PetscRandomDestroy(&rctx));

        std::size_t flops = dgop.flops_apply();
        auto flops_global = reduce_number(flops);

        constexpr int nrepeat = 100;
        Stopwatch sw;
        sw.start();
        for (int i = 0; i < nrepeat; ++i) {
            MatMult(A->mat(), x, y);
        }
        auto time = sw.stop();
        if (rank == 0) {
            time /= nrepeat;
            std::cout << "Shell time: " << time << " s" << std::endl;
            std::cout << "Shell flops: " << flops_global << std::endl;
            std::cout << "Shell GFLOPS: " << flops_global / time * 1e-9 << std::endl;
        }
        CHKERRTHROW(VecDestroy(&x));
        CHKERRTHROW(VecDestroy(&y));
    }

    std::size_t num_dofs_domain = reduce_number(dgop.number_of_local_dofs());

    double local_mesh_size = cl->local_mesh_size();
    double mesh_size;
    MPI_Reduce(&local_mesh_size, &mesh_size, 1, mpi_type_t<double>(), MPI_MAX, 0, topo->comm());

    if (rank == 0) {
        std::cout << "DOFs: " << num_dofs_domain << std::endl;
        std::cout << "Mesh size: " << mesh_size << std::endl;
    }

    sw.start();
    auto solver =
        PetscLinearSolver(dgop, cfg.matrix_free, MGConfig(cfg.mg_coarse_level, cfg.mg_strategy),false);
    time = sw.stop();
    if (rank == 0) {
        std::cout << "Assembly: " << time << " s" << std::endl;
    }

    sw.start();
    solver.warmup();
    time = sw.stop();
    if (rank == 0) {
        std::cout << "Solver warmup: " << time << " s" << std::endl;
    }
    
    using local_operator_t = typename decltype(lop)::element_type;
    constexpr std::size_t NumSlipComponents = slip_traits<local_operator_t>::NumSlipComponents;

    auto fault_map = std::make_shared<BoundaryMap>(mesh, BC::Fault, PETSC_COMM_WORLD);
    auto space = RateAndStateBase::Space();
    auto nbf_fault = space.numBasisFunctions();
    auto slip_block_size = nbf_fault * NumSlipComponents;

    auto adapter = AdapterOperator<local_operator_t>(
        lop,
        std::make_unique<Adapter<local_operator_t>>(cl, space, lop->facetQuadratureRule(), cfg.up,
                                                    cfg.ref_normal),
        topo, fault_map);

    // Only local facets get written; shared ones arrive through the scatter.
    auto faultNo2tag = std::vector<long int>(fault_map->local_size());
    for (std::size_t faultNo = 0; faultNo < fault_map->local_size(); ++faultNo) {
        faultNo2tag[faultNo] = topo->info(fault_map->fctNo(faultNo)).facetTag;
    }

    auto S = PetscVector(slip_block_size, fault_map->local_size(), PETSC_COMM_WORLD);
    auto scatter = Scatter(fault_map->scatter_plan());
    auto ghost = scatter.recv_prototype<double>(slip_block_size, ALIGNMENT);


    auto surfacePoints = receiver_surface_points(mesh, *topo, scenario.transform(),
                                                 cfg.receiver_surface_tag);
    auto grid = make_regular_xy_grid(surfacePoints, cfg.receiver_grid_n, PETSC_COMM_WORLD);
    auto receivers = project_grid_to_receiver_surface(grid.xy, surfacePoints, mesh, *topo, cl,
                                                      lop->solution_prototype(1),
                                                      cfg.receiver_surface_tag, PETSC_COMM_WORLD);

    std::vector<double> u_recv;

    std::unique_ptr<GfHDF5Writer> h5;
    if (cfg.output) {
        h5 = std::make_unique<GfHDF5Writer>(*cfg.output, grid, receivers,
                                            DomainDimension - 1, PETSC_COMM_WORLD);
    }
    

    std::size_t numGfs = gfTags.size() * (DomainDimension - 1);
    std::size_t gfNo = 0;

    for (auto gfTag : gfTags) {
        if (h5) {
            h5->begin_source(gfTag);
        }
        for (std::size_t direction = 0; direction < DomainDimension - 1; ++direction) {
            ++gfNo;

            sw.start();

            set_slip_and_rhs(gfTag, direction, faultNo2tag, nbf_fault, S, scatter, ghost, adapter,
                             dgop, solver);
            solver.solve();

            if (!solver.is_converged()) {
                throw std::runtime_error("Solver did not converge for GF " + std::to_string(gfNo));
            }

            auto u = dgop.solution(solver.x(), receivers.elNos);
            evaluate_receiver_displacement(u, receivers, u_recv);

            if (h5) {
                h5->write_direction(direction, receivers, u_recv);
            }

            time = sw.stop();

            if (rank == 0) {
                std::cout << "Computed " << gfNo << "/" << numGfs << " GF on tag " << gfTag
                        << " in direction " << direction << " in " << time << " s" << std::endl;
            }
        }

        if (h5) {
            h5->end_source();
        }
    }

    if (h5) {
        h5->close();
        h5.reset();
    }

    MPI_Barrier(PETSC_COMM_WORLD);

    if (cfg.output && rank == 0) {
        add_hdf5_metadata(*cfg.output + ".h5", gfTags);
    }

    if (cfg.output) {
        auto fault_map = std::make_shared<BoundaryMap>(mesh, BC::Fault, PETSC_COMM_WORLD);
        auto f = fault_angle_function(cl, lop, topo, fault_map, cfg.up, cfg.ref_normal);
        write_fault_vtu(mesh, cl, fault_map->localFctNos(), f, *cfg.output + "_fault_angles");
    }

}

int main(int argc, char** argv) {
    auto affinity = Affinity();

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

    argparse::ArgumentParser program("sTsGF");
    program.add_argument("--petsc").help("PETSc options, must be passed last!");
    program.add_argument("config").help("Configuration file (.toml)");

    auto makePathRelativeToConfig =
        MakePathRelativeToOtherPath([&program]() { return program.get("config"); });

    TableSchema<Config> schema;
    schema.add_value("resolution", &Config::resolution)
        .validator([](auto&& x) { return x > 0; })
        .help("Non-negative resolution parameter");
    schema.add_value("method", &Config::method)
        .converter([](std::string_view value) {
            if (iEquals(value, "ip")) {
                return DGMethod::IP;
            } else if (iEquals(value, "br2")) {
                return DGMethod::BR2;
            } else {
                return DGMethod::Unknown;
            }
        })
        .default_value(DGMethod::IP)
        .validator([](DGMethod const& type) { return type != DGMethod::Unknown; });
    schema.add_value("type", &Config::type)
        .converter([](std::string_view value) {
            if (iEquals(value, "poisson")) {
                return LocalOpType::Poisson;
            } else if (iEquals(value, "elastic") || iEquals(value, "elasticity")) {
                return LocalOpType::Elasticity;
            } else {
                return LocalOpType::Unknown;
            }
        })
        .validator([](LocalOpType const& type) { return type != LocalOpType::Unknown; });
    schema.add_value("lib", &Config::lib)
        .converter(makePathRelativeToConfig)
        .validator(PathExists());
    schema.add_value("scenario", &Config::scenario);
    {
        auto default_ref_normal = std::array<double, DomainDimension>{};
        default_ref_normal[0] = 1.0;
        schema.add_array("ref_normal", &Config::ref_normal)
            .default_value(std::move(default_ref_normal))
            .of_values();
    }
    schema.add_value("matrix_free", &Config::matrix_free).default_value(false);
    schema.add_value("test_matrix_free", &Config::test_matrix_free).default_value(false);
    schema.add_value("mg_coarse_level", &Config::mg_coarse_level).default_value(1);
    schema.add_value("mg_strategy", &Config::mg_strategy)
        .converter([](std::string_view value) {
            if (iEquals(value, "TwoLevel")) {
                return MGStrategy::TwoLevel;
            } else if (iEquals(value, "Logarithmic")) {
                return MGStrategy::Logarithmic;
            } else if (iEquals(value, "Full")) {
                return MGStrategy::Full;
            } else {
                return MGStrategy::Unknown;
            }
        })
        .default_value(MGStrategy::TwoLevel)
        .validator([](MGStrategy const& type) { return type != MGStrategy::Unknown; });
    schema.add_value("profile", &Config::profile)
        .default_value(0)
        .validator([](auto&& x) { return x >= 0; })
        .help("Run static in profile mode. The parameter controls the amount of repetitions.");
    schema.add_value("output", &Config::output).help("Output file name");
    schema.add_value("mesh_file", &Config::mesh_file)
        .converter(makePathRelativeToConfig)
        .validator(PathExists());
    auto& genMeshSchema = schema.add_table("generate_mesh", &Config::generate_mesh);
    GenMeshConfig<DomainDimension>::setSchema(genMeshSchema);

    {
    auto default_up = std::array<double, DomainDimension>{};
    default_up.back() = 1.0;
    schema.add_array("up", &Config::up)
        .default_value(std::move(default_up))
        .of_values()
        .help("Up direction, used to orient strike and dip in the fault basis.");
    }
    schema.add_value("receiver_surface_tag", &Config::receiver_surface_tag)
    .default_value(100001L)
    .help("Facet tag of the surface receivers are projected onto");
    
    schema.add_value("receiver_grid_n", &Config::receiver_grid_n)
        .default_value(std::size_t(100))
        .validator([](auto&& x) { return x >= 2; })
        .help("Number of receiver grid points along the longer horizontal side");

    std::optional<Config> cfg = readFromConfigurationFileAndCmdLine(schema, program, argc, argv);
    if (!cfg) {
        return -1;
    }

    CHKERRQ(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));
    CHKERRQ(register_PCs());
    CHKERRQ(register_KSPs());

    int rank, procs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &procs);

    auto node_mask = affinity.to_string(affinity.worker_mask_on_node(PETSC_COMM_WORLD));
    if (rank == 0) {
        Banner::standard(std::cout, affinity, node_mask);
    }

    std::unique_ptr<GlobalSimplexMesh<DomainDimension>> globalMesh;
    if (cfg->mesh_file) {
        bool ok = false;
        GlobalSimplexMeshBuilder<DomainDimension> builder;
        std::string meshError;
        if (rank == 0) {
            auto [parser, error] =
                MeshParser::createWithValidation<DomainDimension>(*cfg->mesh_file, &builder);
            if (!parser) {
                meshError = error;
            } else {
                ok = parser->parseFile(*cfg->mesh_file);
                if (!ok) {
                    meshError = *cfg->mesh_file + "\n" + std::string(parser->getErrorMessage());
                }
            }
        }
        MPI_Bcast(&ok, 1, MPI_CXX_BOOL, 0, PETSC_COMM_WORLD);
        if (!ok) {
            if (rank == 0) {
                std::cerr << meshError << std::endl;
            }
            PetscFinalize();
            return -1;
        }
        if (ok) {
            globalMesh = builder.create(PETSC_COMM_WORLD);
        }
        if (procs > 1) {
            // ensure initial element distribution for metis
            globalMesh->repartitionByHash();
        }
    } else if (cfg->generate_mesh && cfg->resolution) {
        auto meshGen = cfg->generate_mesh->create(*cfg->resolution, PETSC_COMM_WORLD);
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

    switch (cfg->type) {
    case LocalOpType::Poisson: {
        auto scenario = PoissonScenario(cfg->lib, cfg->scenario, cfg->ref_normal);
        static_problem(*mesh, scenario, *cfg);
        break;
    }
    case LocalOpType::Elasticity: {
        auto scenario = ElasticityScenario(cfg->lib, cfg->scenario, cfg->ref_normal);
        static_problem(*mesh, scenario, *cfg);
        break;
    }
    default:
        std::cerr << "Unknown type. Should be either poisson or elasticity." << std::endl;
        break;
    };

    PetscErrorCode ierr = PetscFinalize();

    return ierr;
}
