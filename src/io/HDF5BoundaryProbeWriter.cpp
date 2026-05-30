#include "HDF5BoundaryProbeWriter.h"
#include "geometry/PointLocator.h"
#include "io/ProbeWriterUtil.h"
#include "util/LinearAllocator.h"

namespace tndm {

template <std::size_t D>
HDF5BoundaryProbeWriter<D>::HDF5BoundaryProbeWriter(std::string_view prefix,
                                                    std::unique_ptr<TableWriter> table_writer,
                                                    std::vector<Probe<D>> const& probes,
                                                    LocalSimplexMesh<D> const& mesh,
                                                    std::shared_ptr<Curvilinear<D>> cl,
                                                    BoundaryMap const& bnd_map, MPI_Comm comm)
    : hdf5_writer_(std::make_unique<HDF5Writer>(prefix, comm)) {

    // Locate probes in the mesh
    auto bpl =
        BoundaryPointLocator<D>(std::make_shared<PointLocator<D>>(cl), mesh, bnd_map.localFctNos());

    std::vector<std::pair<std::size_t, BoundaryPointLocatorResult<D>>> located_probes;
    located_probes.reserve(probes.size());
    for (std::size_t p = 0; p < probes.size(); ++p) {
        auto result = bpl.locate(probes[p].x);
        located_probes.emplace_back(p, result);
    }

    // Clean duplicate probes using MPI communicator from HDF5Writer
    clean_duplicate_probes(probes, located_probes, hdf5_writer_->comm());

    // Collect unique facet numbers
    std::unordered_set<std::size_t> myFctNos;
    for (auto const& [p, result] : located_probes) {
        myFctNos.emplace(result.no);
    }

    // Map facet numbers to boundary numbers
    bndNos_.reserve(myFctNos.size());
    std::unordered_map<std::size_t, std::size_t> fctNo2OutNo;
    std::size_t outNo = 0;
    for (auto const& fctNo : myFctNos) {
        fctNo2OutNo[fctNo] = outNo++;
        bndNos_.emplace_back(bnd_map.bndNo(fctNo));
    }

    // Store probe data
    probes_.reserve(located_probes.size());
    for (auto const& [p, result] : located_probes) {
        probes_.push_back({
            fctNo2OutNo[result.no], // Local facet number
            result.chi,             // Reference coordinates
            result.x,               // Physical coordinates
            probes[p].name          // Probe name
        });
    }
}

template <std::size_t D>
void HDF5BoundaryProbeWriter<D>::initialize_datasets(
    mneme::span<FiniteElementFunction<D - 1>> functions) {

    // Create time dataset
    std::vector<hsize_t> time_dims = {1};
    std::vector<hsize_t> time_max_dims = {H5S_UNLIMITED};
    int extensibleDimensionTimeStep = 0;
    int glueDimensionTimeStep = 0;
    if (timeStepDataset_ == -1) {
        timeStepDataset_ = hdf5_writer_->createExtendibleDataset(
            "time", H5T_NATIVE_DOUBLE, time_dims, time_max_dims, glueDimensionTimeStep, false);
    }

    // Create probe metadata
    std::vector<double> probe_coords;
    std::vector<const char*> probe_names;
    probe_coords.reserve(probes_.size() * D);
    probe_names.reserve(probes_.size());
    int total_length = 0;
    for (const auto& probe : probes_) {
        probe_coords.insert(probe_coords.end(), probe.x.begin(), probe.x.end());
        probe_names.push_back(probe.name.c_str());
        total_length += probe.name.length();
    }
    // Write probe metadata
    hsize_t numElements = probe_coords.size() / D;
    hsize_t glueDimension = 0;
    hsize_t extensibleDimension = 0;
    hsize_t verticesDataset_ = hdf5_writer_->createExtendibleDataset(
        "probes", H5T_IEEE_F64LE, {numElements, 1, D}, {numElements, 1, D}, glueDimension);
    hdf5_writer_->writeToDataset(verticesDataset_, H5T_IEEE_F64LE, 0, probe_coords.data(),
                                 {numElements, 1, D}, glueDimension, extensibleDimension);
    hdf5_writer_->closeDataset(verticesDataset_);
    // TODO: Write probe names
   
    hsize_t numQuantities = functions[0].numQuantities() + 2 * (D - 2);
    // Create extendible dataset (time is extensible dimension)
    std::vector<hsize_t> dims = {
        probes_.size(),
        1,
        numQuantities,
    };
    std::vector<hsize_t> max_dims = {probes_.size(), H5S_UNLIMITED, numQuantities};

    probe_dataset_ =
        hdf5_writer_->createExtendibleDataset("data", H5T_NATIVE_DOUBLE, dims, max_dims, 0);
}

template <std::size_t D> HDF5BoundaryProbeWriter<D>::~HDF5BoundaryProbeWriter() {
    // Close the time dataset
    if (timeStepDataset_ >= 0) {
        hdf5_writer_->closeDataset(timeStepDataset_);
    }
    // Close all probe datasets
    if (probe_dataset_ >= 0) {
        hdf5_writer_->closeDataset(probe_dataset_);
    }
}

template <std::size_t D>
void HDF5BoundaryProbeWriter<D>::write(double time,
                                       mneme::span<FiniteElementFunction<D - 1>> functions,
                                       hsize_t time_step) {

    hdf5_writer_->writeToDataset(timeStepDataset_, H5T_NATIVE_DOUBLE, time_step, &time,
                                 {time_step + 1}, 0, 0, false);
    auto numQuantities = functions[0].numQuantities();
    std::vector<double> values;
    values.reserve(probes_.size() * numQuantities);

    for (const auto& function : functions) {
        for (const auto& probe : probes_) {
            for (std::size_t q = 0; q < function.numQuantities(); ++q) {
                auto result = Managed<Matrix<double>>(function.mapResultInfo(1));
                auto E = function.evaluationMatrix({probe.chi});
                function.map(probe.no, E, result);
                values.push_back(result(0, q));
            }
        }
    }
    hdf5_writer_->writeToDataset(probe_dataset_, H5T_NATIVE_DOUBLE, time_step, values.data(),
                                 {probes_.size(), time_step + 1, numQuantities}, 0, 1);
}

template class HDF5BoundaryProbeWriter<2u>;
template class HDF5BoundaryProbeWriter<3u>;

} // namespace tndm