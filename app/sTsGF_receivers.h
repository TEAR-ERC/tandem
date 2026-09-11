#ifndef STSGF_RECEIVERS_H
#define STSGF_RECEIVERS_H

#include "config.h"
#include "form/DGOperatorTopo.h"
#include "form/FiniteElementFunction.h"
#include "geometry/Curvilinear.h"
#include "io/HDF5Writer.h"
#include "mesh/LocalSimplexMesh.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace tndm {

/*
 * Regular grid in the horizontal plane, before projection onto the surface.
 * xy is stored j outer, i inner, so index = j * nx + i. That matches the HDF5
 * dimension order (y, x, direction, component).
 */
struct ReceiverGrid {
    std::size_t nx;
    std::size_t ny;
    std::vector<double> x;                 // nx
    std::vector<double> y;                 // ny
    std::vector<std::array<double, 2>> xy; // nx * ny
};

/*
 * One receiver owned by this rank.
 *
 * outNo indexes the FiniteElementFunction returned by
 * dgop.solution(x, set.elNos). E is the evaluation matrix at the receiver's
 * reference coordinate; it depends only on that coordinate, so it is built once
 * here rather than on every solve.
 */
struct ReceiverPoint {
    std::size_t gridIndex;                 // index into the global grid
    std::size_t outNo;
    std::array<double, DomainDimension> x; // projected point on the surface
    double dist;                           // grid point to surface distance
    Managed<Matrix<double>> E;
};

struct ReceiverSet {
    std::vector<ReceiverPoint> points; // only the receivers this rank owns
    std::vector<std::size_t> elNos;    // element subset for dgop.solution
    double maxDistance;                // worst projection distance, over all ranks
};

/* Vertices of every local facet carrying receiverSurfaceTag, after the scenario
 * transform. Feeds the grid bounding box and the mean elevation. */
std::vector<std::array<double, DomainDimension>>
receiver_surface_points(LocalSimplexMesh<DomainDimension> const& mesh, DGOperatorTopo const& topo,
                        typename Curvilinear<DomainDimension>::transform_t const& transform,
                        long int receiverSurfaceTag);

/* Bounding box of the receiver surface, reduced across ranks, filled with a
 * regular grid of roughly square cells. N is the point count along the longer
 * side. Throws if N < 2 or the surface has zero extent. */
ReceiverGrid
make_regular_xy_grid(std::vector<std::array<double, DomainDimension>> const& surfacePoints,
                     std::size_t N, MPI_Comm comm);

/* Drop each grid point onto the nearest point of the receiverSurfaceTag facets,
 * then locate that point in the volume so displacement can be evaluated there.
 * Each receiver ends up owned by exactly one rank. */
ReceiverSet project_grid_to_receiver_surface(
    std::vector<std::array<double, 2>> const& receiverXY,
    std::vector<std::array<double, DomainDimension>> const& surfacePoints,
    LocalSimplexMesh<DomainDimension> const& mesh, DGOperatorTopo const& topo,
    std::shared_ptr<Curvilinear<DomainDimension>> const& cl,
    FiniteElementFunction<DomainDimension> const& prototype, long int receiverSurfaceTag,
    MPI_Comm comm);

/* Surface displacement at the receivers this rank owns. Pass the function from
 * dgop.solution(solver.x(), receivers.elNos). Fills
 * receivers.points.size() * DomainDimension values. */
void evaluate_receiver_displacement(FiniteElementFunction<DomainDimension> const& displacement,
                                    ReceiverSet const& receivers, std::vector<double>& out);

/*
 * Owns the HDF5 file and its datasets for one GF library run.
 *
 * The constructor writes x, y, z, direction and component. Then per source tag:
 * begin_source, one write_direction per slip direction, end_source. Every rank
 * writes only the receivers it owns, so there is no gather.
 */
class GfHDF5Writer {
public:
    GfHDF5Writer(std::string const& prefix, ReceiverGrid const& grid, ReceiverSet const& receivers,
                 std::size_t numDirections, MPI_Comm comm);
    ~GfHDF5Writer();

    GfHDF5Writer(GfHDF5Writer const&) = delete;
    GfHDF5Writer& operator=(GfHDF5Writer const&) = delete;

    void begin_source(long int gfTag);
    void write_direction(std::size_t direction, ReceiverSet const& receivers,
                         std::vector<double> const& displacement);
    void end_source();

    /* Closes the coordinate datasets and the file. The destructor calls it, but
     * call it explicitly before add_hdf5_metadata so the file is flushed. */
    void close();

private:
    std::unique_ptr<HDF5Writer> h5_;
    std::size_t nx_;
    std::size_t ny_;
    std::size_t numDirections_;
    hid_t xDset_ = -1;
    hid_t yDset_ = -1;
    hid_t directionDset_ = -1;
    hid_t componentDset_ = -1;
    hid_t sourceDset_ = -1;
};

/* Attach dimension scales so the datasets are self describing. Rank 0 only:
 * H5P_DEFAULT is the serial driver, and calling this on every rank corrupts the
 * file. Barrier and close the writer first. */
void add_hdf5_metadata(std::string const& filename, std::set<long int> const& gfTags);

} // namespace tndm

#endif // STSGF_RECEIVERS_H