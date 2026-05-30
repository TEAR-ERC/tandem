#include "HDF5Writer.h"
#include <algorithm>
#include <hdf5_hl.h>
#include <iostream>

namespace tndm {

HDF5Writer::HDF5Writer(std::string_view filename, MPI_Comm comm) : comm_(comm) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    // Create HDF5 file with parallel access
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm_, MPI_INFO_NULL);
    file_ =
        H5Fcreate((std::string(filename) + ".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    is_open_ = (file_ >= 0);
    if (!is_open_) {
        std::cerr << "Error: Unable to open HDF5 file!" << std::endl;
    }
}

HDF5Writer::~HDF5Writer() {}

hid_t HDF5Writer::createExtendibleDataset(const std::string_view name, hid_t type,
                                          std::vector<hsize_t> dims, std::vector<hsize_t> max_dims,
                                          int glueDimension, bool isDistributed) {
    if (rank_ == 0) {
        std::cout << "Creating dataset: " << name << std::endl;
    }
    if (std::any_of(dims.begin(), dims.end(), [](hsize_t dim) { return dim == 0; })) {
        std::cerr << "Error: Attempting create a dataset with zero chunk size." << std::endl;
    }
    // Compute total number of faults and rank offsets
    auto [totalDataPoints, _] = calculateOffsets(dims[glueDimension]);
    dims[glueDimension] = isDistributed
                              ? totalDataPoints
                              : dims[glueDimension]; // Update the dimension for the dataset
    // Set initial and max dimensions
    max_dims[glueDimension] = isDistributed ? totalDataPoints : max_dims[glueDimension];

    // Define chunking for efficient access
    std::vector<hsize_t> chunk_dims = dims;
    // Create dataspace
    hid_t dataspace = H5Screate_simple(dims.size(), dims.data(), max_dims.data());

    // Set dataset creation properties (chunking enabled)
    hid_t prop = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(prop, dims.size(), chunk_dims.data());

    // Create the dataset
    hid_t dset = H5Dcreate(file_, name.data(), type, dataspace, H5P_DEFAULT, prop, H5P_DEFAULT);
    if (dset < 0 && rank_ == 0) {
        std::cerr << "Error: Failed to create dataset " << name << std::endl;
    }

    // Clean up resources
    H5Pclose(prop);
    H5Sclose(dataspace);
    return dset; // Caller must close later
}

void HDF5Writer::writeToDataset(hid_t dset, hid_t type, hsize_t timestep, const void* data,
                                std::vector<hsize_t> dims, int glueDimension,
                                int extensibleDimension, bool isDistributed) {

    if (std::any_of(dims.begin(), dims.end(), [](hsize_t dim) { return dim == 0; })) {
        std::cerr << "Error: Attempting to write to dataset with zero chunk size." << std::endl;
    }
    hid_t filespace = H5Dget_space(dset);
    int ndims = H5Sget_simple_extent_ndims(filespace);
    std::vector<hsize_t> current_dims(ndims);
    H5Sget_simple_extent_dims(filespace, current_dims.data(), NULL);

    // Make sure dataset is extended before writing
    std::vector<hsize_t> count = dims;
    // Calculate fault offsets for parallel writes
    auto [totalFaults, offset] = calculateOffsets(dims[glueDimension]);
    // Select hyperslab

    std::vector<hsize_t> start(ndims, 0);
    start[glueDimension] = isDistributed ? offset : start[glueDimension];
    if (timestep >= current_dims[extensibleDimension]) {
        // Extend the dataset if the timestep exceeds current dimensions
        count[extensibleDimension] = 1; // Only extend the first dimension
        start[extensibleDimension] = timestep;
        std::vector<hsize_t> new_dims(ndims);
        new_dims = current_dims;
        new_dims[extensibleDimension] = timestep + 1;
        herr_t status = H5Dset_extent(dset, new_dims.data());
        if (status < 0) {
            std::cerr << "Error extending dataset at timestep " << timestep << std::endl;
            return;
        }
        H5Sclose(filespace);
        filespace = H5Dget_space(dset); // Refresh dataset space after extension
        H5Sget_simple_extent_dims(filespace, current_dims.data(), NULL);
    }

    count[glueDimension] = isDistributed ? dims[glueDimension] : count[glueDimension];
    herr_t select_status =
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start.data(), NULL, count.data(), NULL);
    if (select_status < 0) {
        std::cerr << "Error selecting hyperslab for timestep " << timestep << std::endl;
        return;
    }

    // Create memory space matching the data layout
    hid_t memspace = H5Screate_simple(dims.size(), count.data(), NULL);

    // Collective write
    hid_t plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
    herr_t status = H5Dwrite(dset, type, memspace, filespace, plist, data);
    if (status < 0) {
        std::cerr << "Error writing timestep " << timestep << " on rank " << rank_ << std::endl;
    }

    // Cleanup
    H5Pclose(plist);
    H5Sclose(memspace);
    H5Sclose(filespace);
}

void HDF5Writer::closeDataset(hid_t dset) { H5Dclose(dset); }

std::tuple<hsize_t, hsize_t> HDF5Writer::calculateOffsets(hsize_t localElements) {
    hsize_t totalElements = 0;
    MPI_Allreduce(&localElements, &totalElements, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);

    hsize_t offset = 0;
    MPI_Exscan(&localElements, &offset, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);
    if (rank_ == 0)
        offset = 0;

    return {totalElements, offset};
}

} // namespace tndm
