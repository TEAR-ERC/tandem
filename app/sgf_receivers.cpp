#include "sgf_receivers.h"

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

    int rank;
    MPI_Comm_rank(comm, &rank);

    if constexpr (DomainDimension == 2u) {
        /* 2D: N points along x; coordinate 1 is vertical. ny = 1 and y stays empty. */
        double xmin = std::numeric_limits<double>::max();
        double xmax = std::numeric_limits<double>::lowest();
        for (auto const& p : surfacePoints) {
            xmin = std::min(xmin, p[0]);
            xmax = std::max(xmax, p[0]);
        }
        MPI_Allreduce(MPI_IN_PLACE, &xmin, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(MPI_IN_PLACE, &xmax, 1, MPI_DOUBLE, MPI_MAX, comm);

        double const Lx = xmax - xmin;
        if (Lx <= 0.0) {
            throw std::runtime_error("Receiver surface has zero extent in x");
        }
        double const dx = Lx / (N - 1);

        ReceiverGrid grid;
        grid.nx = N;
        grid.ny = 1;
        grid.x.resize(N);
        grid.xy.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            grid.x[i] = xmin + i * dx;
            grid.xy.push_back({grid.x[i], 0.0});
        }

        if (rank == 0) {
            std::cout << "Receiver grid: " << N << " points along x, dx = " << dx << std::endl;
        }
        return grid;
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

    if (rank == 0) {
        std::cout << "Receiver grid: " << nx << " x " << ny << " = " << nx * ny << " points"
                  << ", dx = " << dx << ", dy = " << dy << std::endl;
    }

    return grid;
}

namespace {

/*
 * Vertical projection of a horizontal grid onto the receiverSurfaceTag facets.
 *
 * 3D: a grid point (x, y) belongs to the facet that contains it in map view.
 * 2D: a grid point x belongs to the surface segment whose x-range contains it.
 * The receiver is the point of that facet above or below the grid point. The
 * element's reference coordinate follows from the facet parametrisation, so no
 * search in the volume is needed. Grid points outside the surface footprint get
 * no receiver (NaN in the output). Map-view facet coordinates equal the facet
 * coordinates for straight-sided facets (the Gmsh meshes are linear).
 */
template <std::size_t D>
ReceiverSet project_vertical(std::vector<std::array<double, 2>> const& receiverXY,
                             LocalSimplexMesh<D> const& mesh, DGOperatorTopo const& topo,
                             std::shared_ptr<Curvilinear<D>> const& cl,
                             FiniteElementFunction<D> const& prototype,
                             long int receiverSurfaceTag, MPI_Comm comm) {
    static_assert(D == 2u || D == 3u, "receiver grid projection needs a 2D or 3D domain");
    constexpr std::size_t H = D - 1u; // horizontal (map-view) dimensions

    /* reference-facet corners: the origin and the unit vectors */
    auto refCorners = std::vector<std::array<double, H>>(D, std::array<double, H>{});
    for (std::size_t k = 1; k < D; ++k) {
        refCorners[k][k - 1] = 1.0;
    }

    struct SurfaceFacet {
        std::size_t elNo;
        std::size_t localFaceNo;
        std::array<std::array<double, H>, D> v; // map-view corners, in refCorners order
    };
    std::vector<SurfaceFacet> facets;
    auto const numLocalElements = mesh.elements().localSize();
    auto F = Managed(cl->mapResultInfo(D));
    for (std::size_t fctNo = 0; fctNo < topo.numLocalFacets(); ++fctNo) {
        if (topo.info(fctNo).facetTag != receiverSurfaceTag) {
            continue;
        }
        for (auto elNo : mesh.template upward<D - 1u>(fctNo)) {
            if (elNo >= numLocalElements) {
                continue; // ghost element: the owning rank takes this facet
            }
            auto dws = mesh.template downward<D - 1u, D>(elNo);
            auto localFaceNo = static_cast<std::size_t>(
                std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fctNo)));
            auto corners = cl->facetParam(localFaceNo, refCorners);
            auto E = cl->evaluateBasisAt(corners);
            cl->map(elNo, E, F);
            SurfaceFacet f{elNo, localFaceNo, {}};
            for (std::size_t k = 0; k < D; ++k) {
                for (std::size_t h = 0; h < H; ++h) {
                    f.v[k][h] = F(h, k);
                }
            }
            facets.emplace_back(f);
        }
    }

    /* map-view point location: first local facet containing the grid point */
    constexpr double tol = 1.0e-12;
    struct Hit {
        std::size_t facet;
        std::array<double, H> chi;
    };
    std::vector<std::pair<std::size_t, Hit>> hits;

    if constexpr (D == 2u) {
        /* a 2D surface has few segments: test each one */
        for (std::size_t id = 0; id < receiverXY.size(); ++id) {
            double const q = receiverXY[id][0];
            for (std::size_t k = 0; k < facets.size(); ++k) {
                double const x0 = facets[k].v[0][0], x1 = facets[k].v[1][0];
                if (x1 == x0) {
                    continue; // vertical segment: no map-view extent
                }
                double c = (q - x0) / (x1 - x0);
                if (c >= -tol && c <= 1.0 + tol) {
                    hits.emplace_back(id, Hit{k, {std::clamp(c, 0.0, 1.0)}});
                    break;
                }
            }
        }
    } else {
        /* bins over the map-view bounding boxes of the local facets */
        double xmin = std::numeric_limits<double>::max(),
               xmax = std::numeric_limits<double>::lowest();
        double ymin = xmin, ymax = xmax, size = 0.0;
        for (auto const& f : facets) {
            for (auto const& v : f.v) {
                xmin = std::min(xmin, v[0]);
                xmax = std::max(xmax, v[0]);
                ymin = std::min(ymin, v[1]);
                ymax = std::max(ymax, v[1]);
            }
            size = std::max({size, std::abs(f.v[1][0] - f.v[0][0]), std::abs(f.v[2][0] - f.v[0][0]),
                             std::abs(f.v[1][1] - f.v[0][1]), std::abs(f.v[2][1] - f.v[0][1])});
        }
        double const bin = size > 0.0 ? size : 1.0;
        auto const nbx = facets.empty() ? 1l : static_cast<long>((xmax - xmin) / bin) + 1;
        auto binOf = [&](double x, double y) {
            return static_cast<long>((x - xmin) / bin) + nbx * static_cast<long>((y - ymin) / bin);
        };
        std::unordered_map<long, std::vector<std::size_t>> bins;
        for (std::size_t k = 0; k < facets.size(); ++k) {
            auto const& v = facets[k].v;
            double fx0 = std::min({v[0][0], v[1][0], v[2][0]}), fx1 = std::max({v[0][0], v[1][0], v[2][0]});
            double fy0 = std::min({v[0][1], v[1][1], v[2][1]}), fy1 = std::max({v[0][1], v[1][1], v[2][1]});
            for (auto j = static_cast<long>((fy0 - ymin) / bin); j <= static_cast<long>((fy1 - ymin) / bin); ++j) {
                for (auto i = static_cast<long>((fx0 - xmin) / bin); i <= static_cast<long>((fx1 - xmin) / bin); ++i) {
                    bins[i + nbx * j].push_back(k);
                }
            }
        }

        for (std::size_t id = 0; id < receiverXY.size(); ++id) {
            auto const& q = receiverXY[id];
            if (facets.empty() || q[0] < xmin || q[0] > xmax || q[1] < ymin || q[1] > ymax) {
                continue;
            }
            auto it = bins.find(binOf(q[0], q[1]));
            if (it == bins.end()) {
                continue;
            }
            for (auto k : it->second) {
                auto const& v = facets[k].v;
                double e1x = v[1][0] - v[0][0], e1y = v[1][1] - v[0][1];
                double e2x = v[2][0] - v[0][0], e2y = v[2][1] - v[0][1];
                double det = e1x * e2y - e1y * e2x;
                if (det == 0.0) {
                    continue; // vertical facet: no map-view area
                }
                double px = q[0] - v[0][0], py = q[1] - v[0][1];
                double c1 = (px * e2y - py * e2x) / det;
                double c2 = (e1x * py - e1y * px) / det;
                if (c1 >= -tol && c2 >= -tol && c1 + c2 <= 1.0 + tol) {
                    c1 = std::clamp(c1, 0.0, 1.0);
                    c2 = std::clamp(c2, 0.0, 1.0 - c1);
                    hits.emplace_back(id, Hit{k, {c1, c2}});
                    break;
                }
            }
        }
    }

    /* a point on a facet edge can be found by several ranks: the lowest rank keeps it */
    int rank;
    MPI_Comm_rank(comm, &rank);
    std::vector<int> owner(receiverXY.size(), std::numeric_limits<int>::max());
    for (auto const& [id, hit] : hits) {
        owner[id] = rank;
    }
    MPI_Allreduce(MPI_IN_PLACE, owner.data(), static_cast<int>(owner.size()), MPI_INT, MPI_MIN,
                  comm);

    std::unordered_map<std::size_t, std::size_t> elNo2OutNo;
    ReceiverSet set;
    set.points.reserve(hits.size());
    auto X = Managed(cl->mapResultInfo(1));
    for (auto const& [id, hit] : hits) {
        if (owner[id] != rank) {
            continue;
        }
        auto const& f = facets[hit.facet];
        std::array<double, D> xi = cl->facetParam(f.localFaceNo, hit.chi);
        auto E = cl->evaluateBasisAt({xi});
        cl->map(f.elNo, E, X);
        std::array<double, D> x{};
        for (std::size_t d = 0; d < D; ++d) {
            x[d] = X(d, 0);
        }

        auto e = elNo2OutNo.find(f.elNo);
        if (e == elNo2OutNo.end()) {
            e = elNo2OutNo.emplace(f.elNo, set.elNos.size()).first;
            set.elNos.emplace_back(f.elNo);
        }
        double dist = 0.0;
        if constexpr (D == 2u) {
            dist = std::abs(x[0] - receiverXY[id][0]);
        } else {
            dist = std::hypot(x[0] - receiverXY[id][0], x[1] - receiverXY[id][1]);
        }
        set.points.emplace_back(
            ReceiverPoint{id, e->second, x, dist, prototype.evaluationMatrix({xi})});
    }

    /* horizontal offset of the receiver from its grid point: round-off only */
    double maxDistLocal = 0.0;
    for (auto const& r : set.points) {
        maxDistLocal = std::max(maxDistLocal, r.dist);
    }
    MPI_Allreduce(&maxDistLocal, &set.maxDistance, 1, MPI_DOUBLE, MPI_MAX, comm);
    unsigned long numLocal = set.points.size(), numOnSurface = 0;
    MPI_Allreduce(&numLocal, &numOnSurface, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    if (rank == 0) {
        std::cout << "Receiver projection (vertical): " << numOnSurface << " of "
                  << receiverXY.size() << " grid points on surface " << receiverSurfaceTag << ", "
                  << receiverXY.size() - numOnSurface
                  << " outside it (NaN in the output); max horizontal offset "
                  << set.maxDistance << std::endl;
    }

    return set;
}

} // namespace

ReceiverSet project_grid_to_receiver_surface(
    std::vector<std::array<double, 2>> const& receiverXY,
    LocalSimplexMesh<DomainDimension> const& mesh, DGOperatorTopo const& topo,
    std::shared_ptr<Curvilinear<DomainDimension>> const& cl,
    FiniteElementFunction<DomainDimension> const& prototype, long int receiverSurfaceTag,
    MPI_Comm comm) {
    return project_vertical<DomainDimension>(receiverXY, mesh, topo, cl, prototype,
                                             receiverSurfaceTag, comm);
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
                           ReceiverSet const& receivers,
                           std::vector<std::size_t> const& directions, MPI_Comm comm)
    : h5_(std::make_unique<HDF5Writer>(prefix, comm)), nx_(grid.nx), ny_(grid.ny),
      directions_(directions) {

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

    /* y (3D only; a 2D grid is a line along x) */
    if constexpr (DomainDimension == 3u) {
        yDset_ = h5_->createFixedDataset("y", H5T_IEEE_F64LE, {ny_});
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

    /*
     * Fault-basis slip direction of each slot: 0 = (up x n) x n, 1 = up x n.
     * Only the selected directions get a slot.
     */
    directionDset_ = h5_->createFixedDataset("direction", H5T_STD_I32LE,
                                             {static_cast<hsize_t>(directions_.size())});
    {
        std::vector<hsize_t> coords;
        std::vector<int> values;
        if (rank == 0) {
            for (hsize_t d = 0; d < directions_.size(); ++d) {
                coords.push_back(d);
                values.push_back(static_cast<int>(directions_[d]));
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
     * Surface elevation (the last coordinate). Unlike the grid coordinates it
     * varies over the grid, z = z(y,x) in 3D and z = z(x) in 2D, so each rank
     * writes only the receivers it owns.
     */
    {
        auto zDims = DomainDimension == 3u ? std::vector<hsize_t>{ny_, nx_}
                                           : std::vector<hsize_t>{nx_};
        auto zDset = h5_->createFixedDataset("z", H5T_IEEE_F64LE, zDims,
                                             std::numeric_limits<double>::quiet_NaN());

        std::vector<hsize_t> coords;
        std::vector<double> values;
        coords.reserve(receivers.points.size() * (DomainDimension - 1));
        values.reserve(receivers.points.size());

        for (auto const& r : receivers.points) {
            if constexpr (DomainDimension == 3u) {
                coords.push_back(r.gridIndex / nx_);
            }
            coords.push_back(r.gridIndex % nx_);
            values.push_back(r.x[DomainDimension - 1]);
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
    /* 3D: (y, x, direction, component); 2D: (x, direction, component) */
    auto dims = std::vector<hsize_t>{};
    if constexpr (DomainDimension == 3u) {
        dims.push_back(ny_);
    }
    dims.insert(dims.end(), {nx_, static_cast<hsize_t>(directions_.size()),
                             static_cast<hsize_t>(DomainDimension)});
    sourceDset_ = h5_->createFixedDataset(
        std::to_string(gfTag), H5T_IEEE_F64LE, dims,
        std::numeric_limits<double>::quiet_NaN()); // grid points off the surface stay NaN
}

void GfHDF5Writer::write_direction(std::size_t direction, ReceiverSet const& receivers,
                                   std::vector<double> const& displacement) {
    auto it = std::find(directions_.begin(), directions_.end(), direction);
    if (it == directions_.end()) {
        throw std::logic_error("write_direction: direction " + std::to_string(direction) +
                               " was not selected");
    }
    auto slot = static_cast<hsize_t>(it - directions_.begin());

    std::vector<hsize_t> coords;
    coords.reserve(receivers.points.size() * (DomainDimension + 1) * DomainDimension);

    for (auto const& r : receivers.points) {
        hsize_t j = r.gridIndex / nx_;
        hsize_t i = r.gridIndex % nx_;

        for (hsize_t c = 0; c < DomainDimension; ++c) {
            if constexpr (DomainDimension == 3u) {
                coords.push_back(j);
            }
            coords.push_back(i);
            coords.push_back(slot);
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
    if (yDset_ >= 0) {
        h5_->closeDataset(yDset_);
    }
    h5_->closeDataset(xDset_);
    h5_.reset();
}

void add_hdf5_metadata(std::string const& filename, std::set<long int> const& gfTags) {
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("Could not open HDF5 file for metadata: " + filename);
    }

    /* 3D datasets are (y, x, direction, component), 2D ones (x, direction, component) */
    constexpr bool has_y = DomainDimension == 3u;
    constexpr unsigned xDim = has_y ? 1u : 0u;

    hid_t y = has_y ? H5Dopen(file, "y", H5P_DEFAULT) : -1;
    hid_t x = H5Dopen(file, "x", H5P_DEFAULT);
    hid_t direction = H5Dopen(file, "direction", H5P_DEFAULT);
    hid_t component = H5Dopen(file, "component", H5P_DEFAULT);
    hid_t z = H5Dopen(file, "z", H5P_DEFAULT);

    if (H5DSset_scale(x, "x") < 0)
        throw std::runtime_error("H5DSset_scale(x) failed");
    if (has_y && H5DSset_scale(y, "y") < 0)
        throw std::runtime_error("H5DSset_scale(y) failed");
    if (H5DSset_scale(direction, "direction") < 0)
        throw std::runtime_error("H5DSset_scale(direction) failed");
    if (H5DSset_scale(component, "component") < 0)
        throw std::runtime_error("H5DSset_scale(component) failed");

    if (has_y && H5DSattach_scale(z, y, 0) < 0)
        throw std::runtime_error("attach y to z failed");
    if (H5DSattach_scale(z, x, xDim) < 0)
        throw std::runtime_error("attach x to z failed");

    for (auto gfTag : gfTags) {
        std::string datasetName = std::to_string(gfTag);
        hid_t dset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);

        if (dset < 0)
            throw std::runtime_error("No dataset \"" + datasetName + "\" in " + filename);

        if (has_y && H5DSattach_scale(dset, y, 0) < 0)
            throw std::runtime_error("attach y failed");
        if (H5DSattach_scale(dset, x, xDim) < 0)
            throw std::runtime_error("attach x failed");
        if (H5DSattach_scale(dset, direction, xDim + 1) < 0)
            throw std::runtime_error("attach direction failed");
        if (H5DSattach_scale(dset, component, xDim + 2) < 0)
            throw std::runtime_error("attach component failed");

        H5Dclose(dset);
    }

    H5Dclose(z);
    H5Dclose(component);
    H5Dclose(direction);
    if (has_y) {
        H5Dclose(y);
    }
    H5Dclose(x);
    H5Fclose(file);
}

} // namespace tndm