#include "sTsGF_receivers.h"

#include "geometry/PointLocator.h"
#include "io/Probe.h"
#include "io/ProbeWriterUtil.h"
#include "mesh/MeshData.h"
#include "util/Range.h"

#include <hdf5.h>
#include <hdf5_hl.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace tndm {

std::vector<std::array<double, DomainDimension>>
receiver_surface_points(LocalSimplexMesh<DomainDimension> const& mesh, DGOperatorTopo const& topo,
                        typename Curvilinear<DomainDimension>::transform_t const& transform,
                        long int receiverSurfaceTag) {
    std::set<std::size_t> vertexIds;

    for (std::size_t fctNo = 0; fctNo < topo.numLocalFacets(); ++fctNo) {
        if (topo.info(fctNo).facetTag != receiverSurfaceTag) {
            continue;
        }
        auto ids = mesh.template downward<0, DomainDimension - 1>(fctNo);
        vertexIds.insert(ids.begin(), ids.end());
    }

    auto vertexData = dynamic_cast<VertexData<DomainDimension> const*>(mesh.vertices().data());
    if (!vertexData) {
        throw std::runtime_error("Vertex data not available.");
    }

    std::vector<std::array<double, DomainDimension>> points;
    points.reserve(vertexIds.size());
    for (auto id : vertexIds) {
        points.push_back(transform(vertexData->getVertices()[id]));
    }

    return points;
}

ReceiverGrid
make_regular_xy_grid(std::vector<std::array<double, DomainDimension>> const& surfacePoints,
                     std::size_t N, MPI_Comm comm) {
    if (N < 2) {
        throw std::runtime_error("Receiver grid needs at least 2 points per side");
    }

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

    if (Lx <= 0.0 || Ly <= 0.0) {
        throw std::runtime_error("Receiver surface has zero extent in x or y");
    }

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

    for (std::size_t i = 0; i < nx; ++i) {
        grid.x[i] = xmin + i * dx;
    }
    for (std::size_t j = 0; j < ny; ++j) {
        grid.y[j] = ymin + j * dy;
    }

    /* j outer, i inner: index = j * nx + i, matching HDF5 dims (ny, nx, ...) */
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            grid.xy.push_back({grid.x[i], grid.y[j]});
        }
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::cout << "Receiver grid: " << nx << " x " << ny << " = " << nx * ny << " points"
                  << ", dx = " << dx << ", dy = " << dy << std::endl;
    }

    return grid;
}

ReceiverSet project_grid_to_receiver_surface(
    std::vector<std::array<double, 2>> const& receiverXY,
    std::vector<std::array<double, DomainDimension>> const& surfacePoints,
    LocalSimplexMesh<DomainDimension> const& mesh, DGOperatorTopo const& topo,
    std::shared_ptr<Curvilinear<DomainDimension>> const& cl,
    FiniteElementFunction<DomainDimension> const& prototype, long int receiverSurfaceTag,
    MPI_Comm comm) {

    std::vector<std::size_t> receiverFacets;
    for (std::size_t fctNo = 0; fctNo < topo.numLocalFacets(); ++fctNo) {
        if (topo.info(fctNo).facetTag == receiverSurfaceTag) {
            receiverFacets.push_back(fctNo);
        }
    }

    auto pointLocator = std::make_shared<PointLocator<DomainDimension>>(cl);
    BoundaryPointLocator<DomainDimension> surfaceLocator(pointLocator, mesh, receiverFacets);

    /* Mean elevation, only a starting guess for the nearest-surface search. */
    double zLocal = 0.0;
    for (auto const& p : surfacePoints) {
        zLocal += p[2];
    }
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
        located.emplace_back(i, surfaceLocator.locate(probes[i].x));
    }

    clean_duplicate_probes(probes, located, comm);

    /*
     * The boundary locate returns a point on the surface but a facet number and
     * facet reference coordinates. Displacement is a volume function, so locate
     * that point again in the volume to get an element and a volume xi.
     */
    auto range = Range<std::size_t>(0, mesh.elements().localSize());

    std::unordered_map<std::size_t, std::size_t> elNo2OutNo;
    ReceiverSet set;
    set.points.reserve(located.size());

    double maxDistLocal = 0.0;

    for (auto const& [id, surf] : located) {
        auto vol = pointLocator->locate(surf.x, range.begin(), range.end());

        auto it = elNo2OutNo.find(vol.no);
        if (it == elNo2OutNo.end()) {
            it = elNo2OutNo.emplace(vol.no, set.elNos.size()).first;
            set.elNos.emplace_back(vol.no);
        }

        maxDistLocal = std::max(maxDistLocal, surf.dist);
        set.points.emplace_back(ReceiverPoint{id, it->second, surf.x, surf.dist,
                                              prototype.evaluationMatrix({vol.xi})});
    }

    /*
     * A grid point outside the mesh footprint snaps to the nearest edge facet
     * and produces a plausible looking wrong number. Surface it instead.
     */
    MPI_Allreduce(&maxDistLocal, &set.maxDistance, 1, MPI_DOUBLE, MPI_MAX, comm);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        std::cout << "Receiver projection: max distance to surface " << set.maxDistance
                  << std::endl;
    }

    return set;
}

void evaluate_receiver_displacement(FiniteElementFunction<DomainDimension> const& displacement,
                                    ReceiverSet const& receivers, std::vector<double>& out) {
    out.assign(receivers.points.size() * DomainDimension, 0.0);
    auto result = Managed<Matrix<double>>(displacement.mapResultInfo(1));

    for (std::size_t i = 0; i < receivers.points.size(); ++i) {
        auto const& r = receivers.points[i];
        displacement.map(r.outNo, r.E, result);
        for (std::size_t c = 0; c < DomainDimension; ++c) {
            out[DomainDimension * i + c] = result(0, c);
        }
    }
}

/* ------------------------------------------------------------------------- *
 * GfHDF5Writer
 * ------------------------------------------------------------------------- */

GfHDF5Writer::GfHDF5Writer(std::string const& prefix, ReceiverGrid const& grid,
                           ReceiverSet const& receivers, std::size_t numDirections, MPI_Comm comm)
    : h5_(std::make_unique<HDF5Writer>(prefix, comm)), nx_(grid.nx), ny_(grid.ny),
      numDirections_(numDirections) {

    int rank;
    MPI_Comm_rank(comm, &rank);

    /* x */
    xDset_ = h5_->createFixedDataset("x", H5T_IEEE_F64LE, {nx_});
    {
        std::vector<hsize_t> coords;
        std::vector<double> values;
        if (rank == 0) {
            coords.resize(nx_);
            values = grid.x;
            for (hsize_t i = 0; i < nx_; ++i) {
                coords[i] = i;
            }
        }
        h5_->writeToDatasetPoints(xDset_, H5T_NATIVE_DOUBLE, coords, values.data());
    }

    /* y */
    yDset_ = h5_->createFixedDataset("y", H5T_IEEE_F64LE, {ny_});
    {
        std::vector<hsize_t> coords;
        std::vector<double> values;
        if (rank == 0) {
            coords.resize(ny_);
            values = grid.y;
            for (hsize_t j = 0; j < ny_; ++j) {
                coords[j] = j;
            }
        }
        h5_->writeToDatasetPoints(yDset_, H5T_NATIVE_DOUBLE, coords, values.data());
    }

    /* slip direction index: 0 = dip, 1 = strike */
    directionDset_ =
        h5_->createFixedDataset("direction", H5T_STD_I32LE, {static_cast<hsize_t>(numDirections_)});
    {
        std::vector<hsize_t> coords;
        std::vector<int> values;
        if (rank == 0) {
            for (hsize_t d = 0; d < numDirections_; ++d) {
                coords.push_back(d);
                values.push_back(static_cast<int>(d));
            }
        }
        h5_->writeToDatasetPoints(directionDset_, H5T_NATIVE_INT, coords, values.data());
    }

    /* displacement component index */
    componentDset_ = h5_->createFixedDataset("component", H5T_STD_I32LE,
                                             {static_cast<hsize_t>(DomainDimension)});
    {
        std::vector<hsize_t> coords;
        std::vector<int> values;
        if (rank == 0) {
            for (hsize_t c = 0; c < DomainDimension; ++c) {
                coords.push_back(c);
                values.push_back(static_cast<int>(c));
            }
        }
        h5_->writeToDatasetPoints(componentDset_, H5T_NATIVE_INT, coords, values.data());
    }

    /*
     * Surface elevation. Unlike x and y this varies over the 2D grid, z = z(y,x),
     * so each rank writes only the receivers it owns.
     */
    {
        auto zDset = h5_->createFixedDataset("z", H5T_IEEE_F64LE, {ny_, nx_});

        std::vector<hsize_t> coords;
        std::vector<double> values;
        coords.reserve(receivers.points.size() * 2);
        values.reserve(receivers.points.size());

        for (auto const& r : receivers.points) {
            coords.push_back(r.gridIndex / nx_);
            coords.push_back(r.gridIndex % nx_);
            values.push_back(r.x[2]);
        }

        h5_->writeToDatasetPoints(zDset, H5T_NATIVE_DOUBLE, coords, values.data());
        h5_->closeDataset(zDset);
    }
}

GfHDF5Writer::~GfHDF5Writer() { close(); }

void GfHDF5Writer::begin_source(long int gfTag) {
    if (sourceDset_ >= 0) {
        throw std::runtime_error("begin_source called without end_source");
    }
    sourceDset_ = h5_->createFixedDataset(
        std::to_string(gfTag), H5T_IEEE_F64LE,
        {ny_, nx_, static_cast<hsize_t>(numDirections_), static_cast<hsize_t>(DomainDimension)});
}

void GfHDF5Writer::write_direction(std::size_t direction, ReceiverSet const& receivers,
                                   std::vector<double> const& displacement) {
    std::vector<hsize_t> coords;
    coords.reserve(receivers.points.size() * 4 * DomainDimension);

    for (auto const& r : receivers.points) {
        hsize_t j = r.gridIndex / nx_;
        hsize_t i = r.gridIndex % nx_;

        for (hsize_t c = 0; c < DomainDimension; ++c) {
            coords.push_back(j);
            coords.push_back(i);
            coords.push_back(static_cast<hsize_t>(direction));
            coords.push_back(c);
        }
    }

    h5_->writeToDatasetPoints(sourceDset_, H5T_NATIVE_DOUBLE, coords, displacement.data());
}

void GfHDF5Writer::end_source() {
    if (sourceDset_ >= 0) {
        h5_->closeDataset(sourceDset_);
        sourceDset_ = -1;
    }
}

void GfHDF5Writer::close() {
    if (!h5_) {
        return;
    }
    end_source();
    h5_->closeDataset(componentDset_);
    h5_->closeDataset(directionDset_);
    h5_->closeDataset(yDset_);
    h5_->closeDataset(xDset_);
    h5_.reset();
}

void add_hdf5_metadata(std::string const& filename, std::set<long int> const& gfTags) {
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("Could not open HDF5 file for metadata: " + filename);
    }

    hid_t y = H5Dopen(file, "y", H5P_DEFAULT);
    hid_t x = H5Dopen(file, "x", H5P_DEFAULT);
    hid_t direction = H5Dopen(file, "direction", H5P_DEFAULT);
    hid_t component = H5Dopen(file, "component", H5P_DEFAULT);
    hid_t z = H5Dopen(file, "z", H5P_DEFAULT);

    if (H5DSset_scale(x, "x") < 0)
        throw std::runtime_error("H5DSset_scale(x) failed");
    if (H5DSset_scale(y, "y") < 0)
        throw std::runtime_error("H5DSset_scale(y) failed");
    if (H5DSset_scale(direction, "direction") < 0)
        throw std::runtime_error("H5DSset_scale(direction) failed");
    if (H5DSset_scale(component, "component") < 0)
        throw std::runtime_error("H5DSset_scale(component) failed");

    if (H5DSattach_scale(z, y, 0) < 0)
        throw std::runtime_error("attach y to z failed");
    if (H5DSattach_scale(z, x, 1) < 0)
        throw std::runtime_error("attach x to z failed");

    for (auto gfTag : gfTags) {
        std::string datasetName = std::to_string(gfTag);
        hid_t dset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);

        if (dset < 0)
            throw std::runtime_error("No dataset \"" + datasetName + "\" in " + filename);

        if (H5DSattach_scale(dset, y, 0) < 0)
            throw std::runtime_error("attach y failed");
        if (H5DSattach_scale(dset, x, 1) < 0)
            throw std::runtime_error("attach x failed");
        if (H5DSattach_scale(dset, direction, 2) < 0)
            throw std::runtime_error("attach direction failed");
        if (H5DSattach_scale(dset, component, 3) < 0)
            throw std::runtime_error("attach component failed");

        H5Dclose(dset);
    }

    H5Dclose(z);
    H5Dclose(component);
    H5Dclose(direction);
    H5Dclose(y);
    H5Dclose(x);
    H5Fclose(file);
}

} // namespace tndm