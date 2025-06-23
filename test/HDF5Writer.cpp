#include "io/HDF5Writer.h"
#include "doctest.h"

#include <filesystem>
#include <mpi.h>
#include <string>
#include <vector>

using namespace tndm;

TEST_CASE("HDF5Writer basic functionality") {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::string filename = "test_output";
    SUBCASE("Initialize HDF5Writer") {
        HDF5Writer writer(filename, comm);
        CHECK(writer.file() >= 0);

        SUBCASE("Create and write extendible dataset") {
            std::vector<hsize_t> initial_dims = {1, 3}; // timestep x 3
            std::vector<hsize_t> max_dims = {H5S_UNLIMITED, 3};
            int extensibleDim = 0;
            int glueDim = 0;

            hid_t dset = writer.createExtendibleDataset(
                "time_series", H5T_NATIVE_DOUBLE, initial_dims, max_dims, extensibleDim, false);
            CHECK(dset >= 0);

            std::vector<double> timestep0 = {0.1, 0.2, 0.3};
            std::vector<hsize_t> write_dims = {1, 3};

            writer.writeToDataset(dset, H5T_NATIVE_DOUBLE, 0, timestep0.data(), write_dims, glueDim,
                                  extensibleDim, false);

            std::vector<double> timestep1 = {1.1, 1.2, 1.3};
            writer.writeToDataset(dset, H5T_NATIVE_DOUBLE, 1, timestep1.data(), write_dims, glueDim,
                                  extensibleDim, false);

            writer.closeDataset(dset);
        }
    }

    SUBCASE("File exists after closing") {
        if (rank == 0) {
            CHECK(std::filesystem::exists(filename + ".h5"));
        }
    }
}
