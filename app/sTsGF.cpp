#include "common/Banner.h"
#include "common/CmdLine.h"
#include "common/ElasticityScenario.h"
#include "common/MGConfig.h"
#include "common/MeshConfig.h"
#include "common/PetscLinearSolver.h"
#include "common/PetscUtil.h"
#include "common/Type.h"
#include "config.h"
#include "form/DGCurvilinearCommon.h"
#include "mesh/LocalSimplexMesh.h"
#include "pc/register.h"

#include "form/DGOperator.h"
#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "geometry/PointLocator.h"
#include "io/GlobalSimplexMeshBuilder.h"
#include "io/HDF5Writer.h"
#include "io/MeshParser.h"
#include "io/Probe.h"
#include "io/ProbeWriterUtil.h"
#include "io/VTUAdapter.h"
#include "io/VTUWriter.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/MeshData.h"
#include "parallel/Affinity.h"
#include "tensor/Managed.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"
#include "util/Stopwatch.h"

#include <argparse.hpp>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/PetscVector.h"

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
    std::size_t receiver_grid_resolution;
};

struct ReceiverPoint {
    std::size_t gridIndex;
    std::array<double, DomainDimension> x;
};

struct ReceiverGrid {
    std::size_t nx;
    std::size_t ny;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<std::array<double, 2>> xy;
};

template <class DGOp>
bool solve_source(DGOp& dgop, PetscLinearSolver& solver, PetscVector& b, std::set<long int> const& activeTags, long int direction) {
    b.set_zero();
    dgop.rhs(b, activeTags, direction);
    CHKERRTHROW(KSPSolve(solver.ksp(), b.vec(), solver.x().vec()));
    return solver.is_converged();
}

template <class DGOp, class CurvilinearType>
void write_vtu(DGOp& dgop, PetscLinearSolver& solver, std::shared_ptr<CurvilinearType> const& cl,
               std::string const& filename, bool writeDisplacement = true,
               bool writeParameters = true) {
    VTUWriter<DomainDimension> writer(PolynomialDegree, true, PETSC_COMM_WORLD);
    auto adapter = CurvilinearVTUAdapter(cl, dgop.num_local_elements());
    auto& piece = writer.addPiece(adapter);

    if (writeDisplacement) {
        auto numeric = dgop.solution(solver.x());
        piece.addPointData(numeric);
        piece.addJacobianData(numeric, adapter);
    }

    if (writeParameters) {
        auto coeffs = dgop.params();
        piece.addPointData(coeffs);
    }

    writer.write(filename);
}

template <class MeshType, class TopoType, class TransformType>
std::vector<std::array<double, DomainDimension>>
get_natural_points(MeshType const& mesh, TopoType const& topo, TransformType const& transform) {
    std::set<std::size_t> vertexIds;

    for (std::size_t fctNo = 0; fctNo < topo.numLocalFacets(); ++fctNo) {
        if (topo.info(fctNo).facetTag != RECEIVER_SURFACE) continue;
        auto ids = mesh.template downward<0, DomainDimension - 1>(fctNo);
        vertexIds.insert(ids.begin(), ids.end());
    }

    auto vertexData = dynamic_cast<VertexData<DomainDimension> const*>(mesh.vertices().data());
    if (!vertexData) throw std::runtime_error("Vertex data not available.");

    std::vector<std::array<double, DomainDimension>> points;
    for (auto id : vertexIds) points.push_back(transform(vertexData->getVertices()[id]));

    return points;
}

template <class DisplacementType, class ReceiverLocationType>
std::vector<double>
evaluate_receiver_displacement(DisplacementType& displacement,
                               std::vector<ReceiverLocationType> const& receiverLocations) {
    std::vector<double> receiverDisplacement(receiverLocations.size() * 3);

    for (std::size_t i = 0; i < receiverLocations.size(); ++i) {
        auto const& receiver = receiverLocations[i];

        auto result = Managed<Matrix<double>>(displacement.mapResultInfo(1));
        auto E = displacement.evaluationMatrix({receiver.xi});

        displacement.map(receiver.element, E, result);

        receiverDisplacement[3 * i + 0] = result(0, 0);
        receiverDisplacement[3 * i + 1] = result(0, 1);
        receiverDisplacement[3 * i + 2] = result(0, 2);
    }

    return receiverDisplacement;
}

std::vector<ReceiverPoint>
project_grid_to_receiver_surface(
    std::vector<std::array<double, 2>> const& receiverXY,
    std::vector<std::array<double, DomainDimension>> const& surfacePoints,
    LocalSimplexMesh<DomainDimension> const& mesh,
    DGOperatorTopo const& topo,
    std::shared_ptr<Curvilinear<DomainDimension>> const& cl,
    MPI_Comm comm) {

    std::vector<std::size_t> receiverFacets;

    for (std::size_t fctNo = 0; fctNo < topo.numLocalFacets(); ++fctNo) {
        if (topo.info(fctNo).facetTag == RECEIVER_SURFACE) receiverFacets.push_back(fctNo);
    }

    auto pointLocator = std::make_shared<PointLocator<DomainDimension>>(cl);
    BoundaryPointLocator<DomainDimension> surfaceLocator(pointLocator, mesh, receiverFacets);

    double zLocal = 0.0;
    for (auto const& p : surfacePoints) zLocal += p[2];

    double nLocal = static_cast<double>(surfacePoints.size());
    double zSum = 0.0;
    double nSum = 0.0;

    MPI_Allreduce(&zLocal, &zSum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&nLocal, &nSum, 1, MPI_DOUBLE, MPI_SUM, comm);

    double zMean = zSum / nSum;

    std::vector<Probe<DomainDimension>> probes;
    probes.reserve(receiverXY.size());

    for (std::size_t i = 0; i < receiverXY.size(); ++i) {
        Probe<DomainDimension> probe;
        probe.name = std::to_string(i);
        probe.x = {receiverXY[i][0], receiverXY[i][1], zMean};
        probes.push_back(probe);
    }

    std::vector<std::pair<std::size_t, BoundaryPointLocatorResult<DomainDimension>>> located;
    located.reserve(probes.size());

    for (std::size_t i = 0; i < probes.size(); ++i) {
        auto result = surfaceLocator.locate(probes[i].x);
        located.emplace_back(i, result);
    }

    clean_duplicate_probes(probes, located, comm);

    std::vector<ReceiverPoint> receiverPoints;
    receiverPoints.reserve(located.size());

    for (auto const& [id, result] : located) receiverPoints.push_back({id, result.x});

    return receiverPoints;
}

ReceiverGrid
make_regular_xy_grid(std::vector<std::array<double, DomainDimension>> const& surfacePoints,
                     std::size_t N,
                     MPI_Comm comm) {

    double xminLocal = std::numeric_limits<double>::max();
    double xmaxLocal = std::numeric_limits<double>::lowest();
    double yminLocal = std::numeric_limits<double>::max();
    double ymaxLocal = std::numeric_limits<double>::lowest();

    for (auto const& p : surfacePoints) {
        xminLocal = std::min(xminLocal, p[0]);
        xmaxLocal = std::max(xmaxLocal, p[0]);
        yminLocal = std::min(yminLocal, p[1]);
        ymaxLocal = std::max(ymaxLocal, p[1]);
    }

    double xmin, xmax, ymin, ymax;

    MPI_Allreduce(&xminLocal, &xmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&xmaxLocal, &xmax, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&yminLocal, &ymin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&ymaxLocal, &ymax, 1, MPI_DOUBLE, MPI_MAX, comm);

    double Lx = xmax - xmin;
    double Ly = ymax - ymin;

    std::size_t nx, ny;
    double dx, dy;

    if (Lx >= Ly) {
        nx = N;
        dx = Lx / (nx - 1);

        ny = std::max<std::size_t>(2, std::llround(Ly / dx) + 1);
        dy = Ly / (ny - 1);
    } else {
        ny = N;
        dy = Ly / (ny - 1);

        nx = std::max<std::size_t>(2, std::llround(Lx / dy) + 1);
        dx = Lx / (nx - 1);
    }

    ReceiverGrid grid;

    grid.nx = nx;
    grid.ny = ny;

    grid.x.resize(nx);
    grid.y.resize(ny);
    grid.xy.reserve(nx * ny);

    for (std::size_t i = 0; i < nx; ++i) grid.x[i] = xmin + i * dx;
    for (std::size_t j = 0; j < ny; ++j) grid.y[j] = ymin + j * dy;

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            grid.xy.push_back({grid.x[i], grid.y[j]});
        }
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::cout << "Receiver grid: "
                  << nx << " x " << ny
                  << " = " << nx * ny << " points"
                  << ", dx = " << dx
                  << ", dy = " << dy
                  << std::endl;
    }

    return grid;
}

template <class Scenario>
void static_problem(LocalSimplexMesh<DomainDimension> const& mesh,
                    Scenario const& scenario,
                    Config const& cfg) {

    tndm::Stopwatch sw;
    double time;

    PetscLogStage solve;
    PetscLogStageRegister("solve", &solve);

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    auto cl = std::make_shared<Curvilinear<DomainDimension>>(
        mesh, scenario.transform(), PolynomialDegree);

    auto lop = scenario.make_local_operator(cl, cfg.method);
    auto topo = std::make_shared<DGOperatorTopo>(mesh, PETSC_COMM_WORLD);

    /*
     * Construct regular receiver grid and project it to the
     * physical receiver surface.
     */
    auto naturalPoints = get_natural_points(mesh, *topo, scenario.transform());

    auto receiverGrid =
        make_regular_xy_grid(naturalPoints, cfg.receiver_grid_resolution, topo->comm());

    auto receiverPoints =
        project_grid_to_receiver_surface(
            receiverGrid.xy,
            naturalPoints,
            mesh,
            *topo,
            cl,
            topo->comm());

    std::size_t localReceiverCount = receiverPoints.size();
    std::size_t globalReceiverCount = 0;

    MPI_Reduce(
        &localReceiverCount,
        &globalReceiverCount,
        1,
        mpi_type_t<std::size_t>(),
        MPI_SUM,
        0,
        topo->comm());

    if (rank == 0) {
        std::cout << "Expected receiver points: "
                  << receiverGrid.nx * receiverGrid.ny
                  << std::endl;

        std::cout << "Actual receiver points:   "
                  << globalReceiverCount
                  << std::endl;
    }

    /*
     * Locate each locally owned receiver point in the DG volume mesh.
     */
    PointLocator<DomainDimension> receiverLocator(cl);

    struct ReceiverLocation {
        std::size_t gridIndex;
        std::size_t element;
        std::array<double, DomainDimension> xi;
    };

    std::vector<ReceiverLocation> receiverLocations;
    receiverLocations.reserve(receiverPoints.size());

    for (auto const& point : receiverPoints) {
        auto result = receiverLocator.locate(point.x);

        if (result.no == std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("Could not locate receiver point.");
        }

        receiverLocations.push_back({
            point.gridIndex,
            result.no,
            result.xi
        });
    }

    /*
     * Find Green-function source tags.
     */
    std::set<long int> sourceTags;
    std::set<long int> alwaysActiveTags;


    for (std::size_t fctNo = 0; fctNo < topo->numLocalFacets(); ++fctNo) {
    auto const& info = topo->info(fctNo);

    if (info.facetTag >= MIN_GF && info.facetTag <= MAX_GF) sourceTags.insert(info.facetTag);
    else alwaysActiveTags.insert(info.facetTag);
}



    /*
     * Build global source-tag list so every MPI rank enters the same
     * sequence of collective KSP solves.
     */
    int mpiSize;
    MPI_Comm_size(topo->comm(), &mpiSize);

    std::vector<long int> localTags(sourceTags.begin(), sourceTags.end());
    int nLocal = localTags.size();

    std::vector<int> counts(mpiSize);

    MPI_Allgather(&nLocal,1,MPI_INT,counts.data(),1,MPI_INT,topo->comm());

    std::vector<int> displs(mpiSize, 0);

    for (int i = 1; i < mpiSize; ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }

    std::vector<long int> allTags(displs.back() + counts.back());

    MPI_Allgatherv(localTags.data(),nLocal,MPI_LONG,allTags.data(),counts.data(),displs.data(), MPI_LONG,topo->comm());

    sourceTags = std::set<long int>(
        allTags.begin(),
        allTags.end());

    auto dgop = DGOperator(topo, std::move(lop));

    const auto reduce_number = [&topo](std::size_t number) {
        std::size_t number_global;

        MPI_Reduce(&number,&number_global, 1,mpi_type_t<std::size_t>(),MPI_SUM, 0,topo->comm());

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

            std::cout << "Shell time: "
                      << time << " s"
                      << std::endl;

            std::cout << "Shell flops: "
                      << flops_global
                      << std::endl;

            std::cout << "Shell GFLOPS: "
                      << flops_global / time * 1e-9
                      << std::endl;
        }

        CHKERRTHROW(VecDestroy(&x));
        CHKERRTHROW(VecDestroy(&y));
    }

    std::size_t num_dofs_domain =
        reduce_number(dgop.number_of_local_dofs());

    double local_mesh_size = cl->local_mesh_size();
    double mesh_size;

    MPI_Reduce(
        &local_mesh_size,
        &mesh_size,
        1,
        mpi_type_t<double>(),
        MPI_MAX,
        0,
        topo->comm());

    if (rank == 0) {
        std::cout << "DOFs: "
                  << num_dofs_domain
                  << std::endl;

        std::cout << "Mesh size: "
                  << mesh_size
                  << std::endl;
    }

    auto solver =
        PetscLinearSolver(
            dgop,
            cfg.matrix_free,
            MGConfig(
                cfg.mg_coarse_level,
                cfg.mg_strategy));

    sw.start();

    solver.warmup();

    time = sw.stop();

    if (rank == 0) {
        std::cout << "Solver warmup: "
                  << time << " s"
                  << std::endl;
    }

    PetscVector b(
        dgop.block_size(),
        topo->numLocalElements(),
        topo->comm());

    std::size_t sourceNumber = 0;
    std::size_t numSources = sourceTags.size();

    std::unique_ptr<HDF5Writer> h5;

    /*
     * Coordinate datasets stay open while source datasets are written
     * because every source dataset attaches to these dimension scales.
     */
    hid_t xDset = -1;
    hid_t yDset = -1;
    hid_t componentDset = -1;

    if (cfg.output) {
        h5 = std::make_unique<HDF5Writer>(
            *cfg.output,
            topo->comm());

        /*
         * ------------------------------------------------------------
         * X coordinate
         * ------------------------------------------------------------
         */
        xDset = h5->createFixedDataset(
            "x",
            H5T_IEEE_F64LE,
            {receiverGrid.nx});

        H5DSset_scale(xDset, "x");

        std::vector<hsize_t> xCoordinates;
        std::vector<double> xValues;

        if (rank == 0) {
            xCoordinates.resize(receiverGrid.nx);
            xValues = receiverGrid.x;

            for (hsize_t i = 0; i < receiverGrid.nx; ++i) {
                xCoordinates[i] = i;
            }
        }

        h5->writeToDatasetPoints(
            xDset,
            H5T_NATIVE_DOUBLE,
            xCoordinates,
            xValues.data());

        /*
         * ------------------------------------------------------------
         * Y coordinate
         * ------------------------------------------------------------
         */
        yDset = h5->createFixedDataset(
            "y",
            H5T_IEEE_F64LE,
            {receiverGrid.ny});

        H5DSset_scale(yDset, "y");

        std::vector<hsize_t> yCoordinates;
        std::vector<double> yValues;

        if (rank == 0) {
            yCoordinates.resize(receiverGrid.ny);
            yValues = receiverGrid.y;

            for (hsize_t j = 0; j < receiverGrid.ny; ++j) {
                yCoordinates[j] = j;
            }
        }

        h5->writeToDatasetPoints(
            yDset,
            H5T_NATIVE_DOUBLE,
            yCoordinates,
            yValues.data());

        /*
         * ------------------------------------------------------------
         * Displacement component coordinate
         * ------------------------------------------------------------
         */
        componentDset = h5->createFixedDataset(
            "component",
            H5T_STD_I32LE,
            {3});

        H5DSset_scale(componentDset, "component");

        std::vector<hsize_t> componentCoordinates;
        std::vector<int> componentValues;

        if (rank == 0) {
            componentCoordinates = {0, 1, 2};
            componentValues = {0, 1, 2};
        }

        h5->writeToDatasetPoints(
            componentDset,
            H5T_NATIVE_INT,
            componentCoordinates,
            componentValues.data());

        /*
         * ------------------------------------------------------------
         * Z coordinate / surface elevation
         *
         * Unlike x and y, z varies over the 2-D surface grid:
         *
         *     z = z(y,x)
         *
         * Each MPI rank writes only the receiver points it owns.
         * ------------------------------------------------------------
         */
        auto zDset = h5->createFixedDataset(
            "z",
            H5T_IEEE_F64LE,
            {receiverGrid.ny, receiverGrid.nx});

        std::vector<hsize_t> zCoordinates;
        std::vector<double> zValues;

        zCoordinates.reserve(receiverPoints.size() * 2);
        zValues.reserve(receiverPoints.size());

        for (auto const& receiver : receiverPoints) {
            hsize_t j = receiver.gridIndex / receiverGrid.nx;
            hsize_t i = receiver.gridIndex % receiverGrid.nx;

            zCoordinates.push_back(j);
            zCoordinates.push_back(i);

            zValues.push_back(receiver.x[2]);
        }

        h5->writeToDatasetPoints(
            zDset,
            H5T_NATIVE_DOUBLE,
            zCoordinates,
            zValues.data());

        H5DSattach_scale(zDset, yDset, 0);
        H5DSattach_scale(zDset, xDset, 1);

        h5->closeDataset(zDset);
    }

    /*
     * ================================================================
     * Green-function source loop
     * ================================================================
     */
    for (auto sourceTag : sourceTags) {
        auto activeTags = alwaysActiveTags;
        activeTags.insert(sourceTag);
        sw.start();

        for (long int direction = 0; direction < DomainDimension - 1; ++direction) {
        bool converged = solve_source(dgop, solver, b, activeTags, direction);
            
        }
        time = sw.stop();

        if (!converged) {
            if (rank == 0) {
                std::cout
                    << "Source "
                    << sourceNumber + 1
                    << " / "
                    << numSources
                    << " failed to converge."
                    << std::endl;
            }

            ++sourceNumber;
            continue;
        }

        if (rank == 0) {
            std::cout
                << "Solved source "
                << sourceNumber + 1
                << " / "
                << numSources
                << " in "
                << time
                << " s"
                << std::endl;
        }
        if (cfg.output) write_vtu(dgop, solver, cl, *cfg.output + "_" + std::to_string(sourceTag), true, true);
        auto displacement =
            dgop.solution(solver.x());

        auto receiverDisplacement =
            evaluate_receiver_displacement(
                displacement,
                receiverLocations);

        /*
         * Write one global [ny,nx,3] dataset for this source.
         */
        if (cfg.output) {
            std::string datasetName =
                std::to_string(sourceTag);

            auto dset =h5->createFixedDataset(datasetName,H5T_IEEE_F64LE,{receiverGrid.ny,receiverGrid.nx,3});
            
            std::vector<hsize_t> coordinates;

            /*
             * Three HDF5 coordinates for every scalar:
             *
             *     (j,i,0)
             *     (j,i,1)
             *     (j,i,2)
             *
             * for each receiver.
             */
            coordinates.reserve(
                receiverLocations.size() * 9);

            for (auto const& receiver : receiverLocations) {
                hsize_t j =
                    receiver.gridIndex /
                    receiverGrid.nx;

                hsize_t i =
                    receiver.gridIndex %
                    receiverGrid.nx;

                for (hsize_t c = 0; c < 3; ++c) {
                    coordinates.push_back(j);
                    coordinates.push_back(i);
                    coordinates.push_back(c);
                }
            }

            h5->writeToDatasetPoints(
                dset,
                H5T_NATIVE_DOUBLE,
                coordinates,
                receiverDisplacement.data());

            /*
             * Tell HDF5/xarray what the three dimensions mean.
             */
            H5DSattach_scale(dset, yDset, 0);
            H5DSattach_scale(dset, xDset, 1);
            H5DSattach_scale(dset, componentDset, 2);

            h5->closeDataset(dset);
        }

        ++sourceNumber;
    }

    /*
     * Coordinate datasets must remain open until all source datasets
     * have had their dimension scales attached.
     */
    if (cfg.output) {
        h5->closeDataset(xDset);
        h5->closeDataset(yDset);
        h5->closeDataset(componentDset);
    }

    if (cfg.output) write_vtu(dgop, solver, cl, *cfg.output, false, true);

    if (cfg.output) h5.reset();

    MPI_Barrier(topo->comm());

    if (cfg.output && rank == 0) {
        std::string h5Filename = *cfg.output + ".h5";

        hid_t file = H5Fopen(h5Filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        hid_t x = H5Dopen(file, "x", H5P_DEFAULT);
        hid_t y = H5Dopen(file, "y", H5P_DEFAULT);
        hid_t component = H5Dopen(file, "component", H5P_DEFAULT);
        hid_t z = H5Dopen(file, "z", H5P_DEFAULT);

        H5DSset_scale(x, "x");
        H5DSset_scale(y, "y");
        H5DSset_scale(component, "component");

        H5DSattach_scale(z, y, 0);
        H5DSattach_scale(z, x, 1);

        for (auto sourceTag : sourceTags) {
            std::string datasetName = std::to_string(sourceTag);
            hid_t dset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);

            H5DSattach_scale(dset, y, 0);
            H5DSattach_scale(dset, x, 1);
            H5DSattach_scale(dset, component, 2);

            H5Dclose(dset);
        }

        H5Dclose(z);
        H5Dclose(component);
        H5Dclose(y);
        H5Dclose(x);
        H5Fclose(file);
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

    argparse::ArgumentParser program("static");

    program.add_argument("--petsc")
        .help("PETSc options, must be passed last!");

    program.add_argument("config")
        .help("Configuration file (.toml)");

    auto makePathRelativeToConfig =
        MakePathRelativeToOtherPath(
            [&program]() {
                return program.get("config");
            });

    TableSchema<Config> schema;

    schema.add_value(
        "resolution",
        &Config::resolution)
        .validator([](auto&& x) {
            return x > 0;
        })
        .help("Non-negative resolution parameter");

    schema.add_value(
        "method",
        &Config::method)
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
        .validator([](DGMethod const& type) {
            return type != DGMethod::Unknown;
        });

    schema.add_value(
        "type",
        &Config::type)
        .converter([](std::string_view value) {
            if (iEquals(value, "elastic") ||
                iEquals(value, "elasticity")) {
                return LocalOpType::Elasticity;
            } else {
                return LocalOpType::Unknown;
            }
        })
        .validator([](LocalOpType const& type) {
            return type == LocalOpType::Elasticity;
        });

    schema.add_value(
        "lib",
        &Config::lib)
        .converter(makePathRelativeToConfig)
        .validator(PathExists());

    schema.add_value(
        "scenario",
        &Config::scenario);

    {
        auto default_ref_normal =
            std::array<double, DomainDimension>{};

        default_ref_normal[0] = 1.0;

        schema.add_array(
            "ref_normal",
            &Config::ref_normal)
            .default_value(
                std::move(default_ref_normal))
            .of_values();
    }

    schema.add_value(
        "matrix_free",
        &Config::matrix_free)
        .default_value(false);

    schema.add_value(
        "test_matrix_free",
        &Config::test_matrix_free)
        .default_value(false);

    schema.add_value(
        "mg_coarse_level",
        &Config::mg_coarse_level)
        .default_value(1);

    schema.add_value(
        "mg_strategy",
        &Config::mg_strategy)
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
        .validator([](MGStrategy const& type) {
            return type != MGStrategy::Unknown;
        });

    schema.add_value(
        "profile",
        &Config::profile)
        .default_value(0)
        .validator([](auto&& x) {
            return x >= 0;
        })
        .help(
            "Run static in profile mode. "
            "The parameter controls the amount of repetitions.");

    schema.add_value(
        "output",
        &Config::output)
        .help("Output file name");

    schema.add_value(
        "mesh_file",
        &Config::mesh_file)
        .converter(makePathRelativeToConfig)
        .validator(PathExists());

    schema.add_value(
        "receiver_grid_resolution",
        &Config::receiver_grid_resolution)
        .default_value(100)
        .validator([](auto&& x) {
            return x >= 2;
        })
        .help("Number of receiver-grid points along the longest horizontal dimension.");

    auto& genMeshSchema =
        schema.add_table(
            "generate_mesh",
            &Config::generate_mesh);

    GenMeshConfig<DomainDimension>::setSchema(
        genMeshSchema);

    std::optional<Config> cfg =
        readFromConfigurationFileAndCmdLine(
            schema,
            program,
            argc,
            argv);

    if (!cfg) return -1;

    CHKERRQ(
        PetscInitialize(
            &pArgc,
            &pArgv,
            nullptr,
            nullptr));

    CHKERRQ(register_PCs());
    CHKERRQ(register_KSPs());

    int rank, procs;

    MPI_Comm_rank(
        PETSC_COMM_WORLD,
        &rank);

    MPI_Comm_size(
        PETSC_COMM_WORLD,
        &procs);

    auto node_mask =
        affinity.to_string(
            affinity.worker_mask_on_node(
                PETSC_COMM_WORLD));

    if (rank == 0) {
        Banner::standard(
            std::cout,
            affinity,
            node_mask);
    }

    std::unique_ptr<
        GlobalSimplexMesh<DomainDimension>>
        globalMesh;

    if (cfg->mesh_file) {
        bool ok = false;

        GlobalSimplexMeshBuilder<DomainDimension>
            builder;

        std::string meshError;

        if (rank == 0) {
            auto [parser, error] =
                MeshParser::createWithValidation<
                    DomainDimension>(
                    *cfg->mesh_file,
                    &builder);

            if (!parser) {
                meshError = error;
            } else {
                ok = parser->parseFile(
                    *cfg->mesh_file);

                if (!ok) {
                    meshError =
                        *cfg->mesh_file +
                        "\n" +
                        std::string(
                            parser->getErrorMessage());
                }
            }
        }

        MPI_Bcast(
            &ok,
            1,
            MPI_CXX_BOOL,
            0,
            PETSC_COMM_WORLD);

        if (!ok) {
            if (rank == 0) {
                std::cerr
                    << meshError
                    << std::endl;
            }

            PetscFinalize();
            return -1;
        }

        globalMesh =
            builder.create(
                PETSC_COMM_WORLD);

        if (procs > 1) {
            globalMesh->repartitionByHash();
        }

    } else if (
        cfg->generate_mesh &&
        cfg->resolution) {

        auto meshGen =
            cfg->generate_mesh->create(
                *cfg->resolution,
                PETSC_COMM_WORLD);

        globalMesh =
            meshGen.uniformMesh();
    }

    if (!globalMesh) {
        std::cerr
            << "You must either provide a valid mesh file "
               "or provide the mesh generation config "
               "(including the resolution parameter)."
            << std::endl;

        PetscFinalize();
        return -1;
    }

    globalMesh->repartition();

    auto mesh =
        globalMesh->getLocalMesh(1);

    switch (cfg->type) {


    case LocalOpType::Elasticity: {
        auto scenario =
            ElasticityScenario(
                cfg->lib,
                cfg->scenario,
                cfg->ref_normal);

        static_problem(
            *mesh,
            scenario,
            *cfg);

        break;
    }

    default:
        std::cerr
            << "Unknown type. Should be either poisson or elasticity."
            << std::endl;

        break;
    }

    PetscErrorCode ierr =
        PetscFinalize();

    return ierr;
}