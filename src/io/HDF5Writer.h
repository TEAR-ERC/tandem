#ifndef HDF5_WRITER_H
#define HDF5_WRITER_H

#include <cstdint>
#include <mpi.h>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#ifdef ENABLE_HDF5
#include <hdf5.h>
#include <hdf5_hl.h>
#else
using hid_t = int64_t;
using hsize_t = uint64_t;
#endif

namespace tndm {
class HDF5Writer {
public:
    HDF5Writer(std::string_view filename, MPI_Comm comm);
    ~HDF5Writer();
    // Returns the dataset ID for later writing
    hid_t createExtendibleDataset(const std::string_view name, hid_t type,
                                  std::vector<hsize_t> dims, std::vector<hsize_t> max_dims,
                                  int extensibleDimension, bool isDistributed = true);

    // Takes the dataset ID as parameter
    void writeToDataset(hid_t dset, hid_t type, hsize_t localElements, const void* data,
                        std::vector<hsize_t> dims, int glueDimension, int extensibleDimension,
                        bool isDistributed = true);

    // Add explicit close function
    void closeDataset(hid_t dset);
    MPI_Comm comm() const { return comm_; }
    hid_t file() const { return file_; }

private:
    hid_t file_;
    MPI_Comm comm_;
    int rank_, size_;
    bool is_open_ = false;
    // Helper to calculate offsets and total elements
    std::tuple<hsize_t, hsize_t> calculateOffsets(hsize_t localElements);
};
} // namespace tndm
#endif
