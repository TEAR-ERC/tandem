#include "HDF5ProbeWriter.h"
#include "geometry/PointLocator.h"
#include "io/ProbeWriterUtil.h"
#include "util/LinearAllocator.h"

namespace tndm {

template <std::size_t D, bool isBoundary>
auto makeLocator(std::shared_ptr<Curvilinear<D>> cl, LocalSimplexMesh<D> const& mesh,
                 BoundaryMap const& bnd_map) {
    using LocatorType = std::conditional_t<isBoundary, BoundaryPointLocator<D>, PointLocator<D>>;

    if constexpr (isBoundary) {
        return LocatorType(std::make_shared<PointLocator<D>>(cl), mesh, bnd_map.localFctNos());
    } else {
        return LocatorType(cl);
    }
}

template <std::size_t D, bool isBoundary>
HDF5ProbeWriter<D, isBoundary>::HDF5ProbeWriter(std::string_view prefix,
                                                std::unique_ptr<TableWriter> table_writer,
                                                std::vector<Probe<D>> const& probes,
                                                LocalSimplexMesh<D> const& mesh,
                                                std::shared_ptr<Curvilinear<D>> cl,
                                                BoundaryMap const& bnd_map, MPI_Comm comm)
    : hdf5_writer_(std::make_unique<HDF5Writer>(prefix, comm)) {

    using ElementFunction = tndm::FiniteElementFunction<isBoundary ? D - 1 : D>;
    using ResultType =
        std::conditional_t<isBoundary, BoundaryPointLocatorResult<D>, PointLocatorResult<D>>;
    auto locator = makeLocator<D, isBoundary>(cl, mesh, bnd_map);
    auto range = Range<std::size_t>(0, mesh.elements().localSize());
    std::vector<std::pair<std::size_t, ResultType>> located_probes;
    located_probes.reserve(probes.size());
    for (std::size_t p = 0; p < probes.size(); ++p) {
        auto result = [&]() {
            if constexpr (isBoundary) {
                return locator.locate(probes[p].x);
            } else {
                return locator.locate(probes[p].x, range.begin(), range.end());
            }
        }();
        located_probes.emplace_back(p, result);
    }

    clean_duplicate_probes(probes, located_probes, hdf5_writer_->comm());

    std::unordered_set<std::size_t> entityNos;
    for (auto const& probe_pair : located_probes) {
        auto const& result = probe_pair.second;
        entityNos.emplace(result.no);
    }

    bndNos_.reserve(entityNos.size());
    std::unordered_map<std::size_t, std::size_t> entityNo2OutNo;
    std::size_t outNo = 0;
    for (auto const& no : entityNos) {
        entityNo2OutNo[no] = outNo++;
        bndNos_.emplace_back(isBoundary ? bnd_map.bndNo(no) : no);
    }

    // Store probe data
    probes_.reserve(located_probes.size());
    for (auto const& probe_pair : located_probes) {
        auto const& p = probe_pair.first;
        auto const& result = probe_pair.second;
        auto const& refCoord = [&]() {
            if constexpr (isBoundary) {
                return result.chi;
            } else {
                return result.xi;
            }
        }();
        probes_.push_back({
            entityNo2OutNo[result.no], // Local facet number
            refCoord,                  // Reference coordinates
            result.x,                  // Physical coordinates
            probes[p].name             // Probe name
        });
    }
}

template <std::size_t D, bool isBoundary>
void HDF5ProbeWriter<D, isBoundary>::initialize_datasets(mneme::span<ElementFunction> functions) {
    // Create probe metadata
    std::vector<char> probe_names;
    std::vector<double> probe_coords;
    int max_length_local = 1;
    for (const auto& probe : probes_) {
        max_length_local = std::max(int(probe.name.length()), max_length_local);
    }
    int max_length_global;
    MPI_Allreduce(&max_length_local, &max_length_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    probe_coords.reserve(probes_.size() * D);
    for (const auto& probe : probes_) {
        probe_coords.insert(probe_coords.end(), probe.x.begin(), probe.x.end());
        std::string padded_name = probe.name;
        padded_name.resize(max_length_local, ' ');
        probe_names.insert(probe_names.end(), padded_name.begin(), padded_name.end());
    }
    hsize_t numProbes = probes_.size();

    // Write probe metadata
    // Write probe vertices
    hsize_t glueDimension = 0;
    hsize_t extensibleDimension = 0;
    std::vector<hsize_t> probe_vertices_dims = {numProbes, 1, D};
    std::vector<hsize_t> probe_vertices_max_dims = {numProbes, 1, D};
    hsize_t verticesDataset_ = hdf5_writer_->createExtendibleDataset(
        "probePositions", H5T_NATIVE_DOUBLE, probe_vertices_dims, probe_vertices_max_dims,
        glueDimension);
    hdf5_writer_->writeToDataset(verticesDataset_, H5T_NATIVE_DOUBLE, 0, probe_coords.data(),
                                 probe_vertices_dims, glueDimension, extensibleDimension);
    hdf5_writer_->closeDataset(verticesDataset_);
    // Write probe names
    auto strtype = H5Tcopy(H5T_C_S1);
    std::vector<hsize_t> probe_names_dims = {numProbes};
    std::vector<hsize_t> probe_names_max_dims = {numProbes};
    H5Tset_size(strtype, max_length_global);  // Set to maximum string length type
    H5Tset_strpad(strtype, H5T_STR_NULLTERM); // Pad with spaces
    hsize_t probeNameDataset_ = hdf5_writer_->createExtendibleDataset(
        "probeNames", strtype, probe_names_dims, probe_names_max_dims, glueDimension);

    hdf5_writer_->writeToDataset(probeNameDataset_, strtype, 0, probe_names.data(),
                                 probe_names_dims, glueDimension, extensibleDimension);
    hdf5_writer_->closeDataset(probeNameDataset_);
    H5Tclose(strtype); // Clean up type
}

template <std::size_t D, bool isBoundary> HDF5ProbeWriter<D, isBoundary>::~HDF5ProbeWriter() {
    // Close the time dataset
    if (timeStepDataset_ >= 0) {
        hdf5_writer_->closeDataset(timeStepDataset_);
    }
    // Close all probe datasets
    if (probe_dataset_ >= 0) {
        hdf5_writer_->closeDataset(probe_dataset_);
    }
}

template <std::size_t D, bool isBoundary>
void HDF5ProbeWriter<D, isBoundary>::write(double time, mneme::span<ElementFunction> functions,
                                           hsize_t time_step) {

    if (timeStepDataset_ == -1) {
        // Create time dataset
        std::vector<hsize_t> time_dims = {1};
        std::vector<hsize_t> time_max_dims = {H5S_UNLIMITED};
        int extensibleDimensionTimeStep = 0;
        int glueDimensionTimeStep = 0;
        bool isDistributed = false;
        timeStepDataset_ = hdf5_writer_->createExtendibleDataset(
            "time", H5T_NATIVE_DOUBLE, time_dims, time_max_dims, glueDimensionTimeStep,
            isDistributed);
    }
    hdf5_writer_->writeToDataset(timeStepDataset_, H5T_NATIVE_DOUBLE, time_step, &time,
                                 {time_step + 1}, 0, 0, false);

    // Write probe data
    auto numQuantities = functions[0].numQuantities();
    auto numProbes = probes_.size();
    std::vector<std::string> probeFields(numQuantities);
    std::vector<char> probe_field_names;
    for (std::size_t q = 0; q < numQuantities; ++q) {
        probeFields[q] = functions[0].name(q);
    }
    int max_length_local = 1;
    for (const auto& field : probeFields) {
        max_length_local = std::max(int(field.length()), max_length_local);
    }
    for (auto& field : probeFields) {
        std::string padded_name = field;
        padded_name.resize(max_length_local, ' ');
        probe_field_names.insert(probe_field_names.end(), padded_name.begin(), padded_name.end());
    }
    int max_length_global = 0;
    MPI_Allreduce(&max_length_local, &max_length_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    // Write probe field names (slip, slip rate, tractions etc.)
    auto strtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(strtype, max_length_global);
    H5Tset_strpad(strtype, H5T_STR_NULLTERM);
    if (!initialized_) {
        // Create time dataset
        probeFieldsDataset_ = hdf5_writer_->createExtendibleDataset(
            "probeFields", strtype, {numQuantities}, {numQuantities}, 0, false);
        hdf5_writer_->writeToDataset(probeFieldsDataset_, strtype, 0, probe_field_names.data(),
                                     {numQuantities}, 0, 0, false);
        hdf5_writer_->closeDataset(probeFieldsDataset_);
        initialized_ = true;
    }

    if (probe_dataset_ == -1) {
        // Create probe dataset
        std::vector<hsize_t> probe_dims = {numProbes, 1, numQuantities};
        std::vector<hsize_t> probe_max_dims = {numProbes, H5S_UNLIMITED, numQuantities};
        int extensibleDimensionProbe = 1;
        int glueDimensionProbe = 0;
        probe_dataset_ = hdf5_writer_->createExtendibleDataset(
            "probeData", H5T_NATIVE_DOUBLE, probe_dims, probe_max_dims, glueDimensionProbe);
    }
    std::vector<double> values;
    values.reserve(numProbes * numQuantities);
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
                                 {numProbes, time_step + 1, numQuantities}, 0, 1);
}

template class HDF5ProbeWriter<2u, true>;
template class HDF5ProbeWriter<3u, true>;
template class HDF5ProbeWriter<2u, false>;
template class HDF5ProbeWriter<3u, false>;

} // namespace tndm