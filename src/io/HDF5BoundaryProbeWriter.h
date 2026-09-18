#ifndef HDF5_BOUNDARYPROBEWRITER_H
#define HDF5_BOUNDARYPROBEWRITER_H

#include "HDF5Writer.h"
#include "form/BoundaryMap.h"
#include "form/FiniteElementFunction.h"
#include "geometry/Curvilinear.h"
#include "io/Probe.h"
#include "mesh/LocalSimplexMesh.h"

#include "form/BoundaryMap.h"
#include "form/FiniteElementFunction.h"
#include "geometry/Curvilinear.h"
#include "io/Probe.h"
#include "io/TableWriter.h"
#include "mesh/LocalSimplexMesh.h"
#include <mneme/span.hpp>
#include <mpi.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tndm {

template <std::size_t D> class HDF5BoundaryProbeWriter {
public:
    HDF5BoundaryProbeWriter(std::string_view prefix, std::unique_ptr<TableWriter> table_writer,
                            std::vector<Probe<D>> const& probes, LocalSimplexMesh<D> const& mesh,
                            std::shared_ptr<Curvilinear<D>> cl, BoundaryMap const& bnd_map,
                            MPI_Comm comm);
    ~HDF5BoundaryProbeWriter();
    void write(double time, mneme::span<FiniteElementFunction<D - 1>> functions, hsize_t timestep);

    auto begin() const { return bndNos_.begin(); }
    auto end() const { return bndNos_.end(); }
    auto num_probes() const { return probes_.size(); }
    std::vector<std::size_t> const& bndNos() const { return bndNos_; }

    struct ProbeData {
        std::size_t no;                // Local facet number
        std::array<double, D - 1> chi; // Reference coordinates
        std::array<double, D> x;       // Physical coordinates
        std::string name;              // Probe name
    };

    void initialize_datasets(mneme::span<FiniteElementFunction<D - 1>> functions);
    void write_probe_metadata();

private:
    std::unique_ptr<HDF5Writer> hdf5_writer_;
    std::vector<std::size_t> bndNos_;
    std::vector<ProbeData> probes_;
    bool initialized_ = false;
    hid_t timeStepDataset_ = -1;
    hid_t probe_dataset_ = -1; 
    hid_t probeFieldsDataset_ = -1; 
};

} // namespace tndm

#endif // HDF5_BOUNDARYPROBEWRITER_H