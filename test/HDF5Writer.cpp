#include "io/HDF5Writer.h"
#include "doctest.h"

#include <filesystem>
#include <mpi.h>
#include <string>
#include <vector>

using namespace tndm;

#ifdef ENABLE_HDF5
TEST_CASE("HDF5Writer - basic functionality") {
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

            writer.writeToDataset(dset, H5T_NATIVE_DOUBLE, 1e-3, timestep0.data(), write_dims,
                                  glueDim, extensibleDim, false);

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

    // Cleanup
    if (rank == 0) {
        std::filesystem::remove("test_output.h5");
    }
    MPI_Barrier(comm); // wait for rank 0 to finish cleanup
}
TEST_CASE("HDF5Writer - file is properly closed after destruction") {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::string filename = "test_close";
    // Create a scope to ensure the HDF5Writer is destroyed
    {
        HDF5Writer writer(filename, comm);
        CHECK(writer.file() >= 0);
    } // destructor fires here

    // After destruction, attempt to open the file
    // If H5Fclose was never called, the file will be corrupt/incomplete
    // and H5Fopen will either fail or return an invalid superblock
    hid_t file = H5Fopen((filename + ".h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(file >= 0);
    if (file >= 0) {
        H5Fclose(file);
    }

    // Cleanup
    if (rank == 0) {
        std::filesystem::remove("test_close.h5");
    }
    MPI_Barrier(comm); // wait for rank 0 to finish cleanup
}

TEST_CASE("HDF5Writer - dataset extends correctly") {
    {
        HDF5Writer writer("test_extend", MPI_COMM_WORLD);
        std::vector<hsize_t> dims = {1, 3};
        std::vector<hsize_t> max_dims = {H5S_UNLIMITED, 3};
        hid_t dset =
            writer.createExtendibleDataset("data", H5T_NATIVE_DOUBLE, dims, max_dims, 0, false);
        for (int t = 0; t < 10; ++t) {
            std::vector<double> data = {t * 1.0, t * 2.0, t * 3.0};
            writer.writeToDataset(dset, H5T_NATIVE_DOUBLE, t, data.data(), {1, 3}, 0, 0, false);
        }
        writer.closeDataset(dset);
    }

    // Should be safe to read back
    hid_t file = H5Fopen("test_extend.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(file >= 0);
    hid_t read_dset = H5Dopen(file, "data", H5P_DEFAULT);
    hid_t space = H5Dget_space(read_dset);
    std::vector<hsize_t> read_dims(2);
    H5Sget_simple_extent_dims(space, read_dims.data(), NULL);
    CHECK(read_dims[0] == 10);
    CHECK(read_dims[1] == 3);
    H5Sclose(space);
    H5Dclose(read_dset);
    H5Fclose(file);

    // Cleanup
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::filesystem::remove("test_extend.h5");
    }
    MPI_Barrier(MPI_COMM_WORLD); // wait for rank 0 to finish cleanup
}

TEST_CASE("HDF5Writer - checkpoint mode appends to existing file") {
    std::string filename = "test_checkpoint";
    std::filesystem::remove(filename + ".h5");

    // Create initial file with 3 timesteps
    {
        HDF5Writer writer(filename, MPI_COMM_WORLD, true); // Enable checkpointing
        std::vector<hsize_t> dims = {1, 2};
        std::vector<hsize_t> max_dims = {H5S_UNLIMITED, 2};
        hid_t dset =
            writer.createExtendibleDataset("data", H5T_NATIVE_DOUBLE, dims, max_dims, 0, false);

        for (int t = 0; t < 3; ++t) {
            std::vector<double> data = {t * 10.0, t * 20.0};
            writer.writeToDataset(dset, H5T_NATIVE_DOUBLE, t, data.data(), {1, 2}, 0, 0, false);
        }
        writer.closeDataset(dset);
    }

    // Reopen with checkpoint and append more timesteps
    {
        HDF5Writer writer(filename, MPI_COMM_WORLD, true); // Enable checkpointing
        hid_t dset = H5Dopen(writer.file(), "data", H5P_DEFAULT);
        CHECK(dset >= 0);

        // Append 2 more timesteps (t=3, t=4)
        for (int t = 3; t < 5; ++t) {
            std::vector<double> data = {t * 10.0, t * 20.0};
            writer.writeToDataset(dset, H5T_NATIVE_DOUBLE, t, data.data(), {1, 2}, 0, 0, false);
        }
        writer.closeDataset(dset);
    }

    // Should have 5 timesteps total
    hid_t file = H5Fopen((filename + ".h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(file >= 0);
    hid_t read_dset = H5Dopen(file, "data", H5P_DEFAULT);
    hid_t space = H5Dget_space(read_dset);
    std::vector<hsize_t> read_dims(2);
    H5Sget_simple_extent_dims(space, read_dims.data(), NULL);
    CHECK(read_dims[0] == 5); // 3 original + 2 appended
    CHECK(read_dims[1] == 2);

    // Verify data values
    std::vector<double> all_data(5 * 2);
    H5Dread(read_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, all_data.data());
    for (int t = 0; t < 5; ++t) {
        CHECK(all_data[t * 2 + 0] == doctest::Approx(t * 10.0));
        CHECK(all_data[t * 2 + 1] == doctest::Approx(t * 20.0));
    }

    H5Sclose(space);
    H5Dclose(read_dset);
    H5Fclose(file);

    // Cleanup
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::filesystem::remove("test_checkpoint.h5");
    }
    MPI_Barrier(MPI_COMM_WORLD); // wait for rank 0 to finish cleanup
}

TEST_CASE("HDF5Writer - checkpoint disabled creates fresh file") {
    std::string filename = "test_no_checkpoint";
    std::filesystem::remove(filename + ".h5");

    // Create file with 3 timesteps
    {
        HDF5Writer writer(filename, MPI_COMM_WORLD, false); // Disable checkpointing
        std::vector<hsize_t> dims = {1, 2};
        std::vector<hsize_t> max_dims = {H5S_UNLIMITED, 2};
        hid_t dset =
            writer.createExtendibleDataset("data", H5T_NATIVE_DOUBLE, dims, max_dims, 0, false);

        for (int t = 0; t < 3; ++t) {
            std::vector<double> data = {t * 10.0, t * 20.0};
            writer.writeToDataset(dset, H5T_NATIVE_DOUBLE, t, data.data(), {1, 2}, 0, 0, false);
        }
        writer.closeDataset(dset);
    }

    // Reopen WITHOUT checkpoint - should OVERWRITE (fresh file)
    {
        HDF5Writer writer(filename, MPI_COMM_WORLD, false); // Disable checkpointing
        std::vector<hsize_t> dims = {1, 2};
        std::vector<hsize_t> max_dims = {H5S_UNLIMITED, 2};
        hid_t dset =
            writer.createExtendibleDataset("data", H5T_NATIVE_DOUBLE, dims, max_dims, 0, false);

        // Write only 2 timesteps (fresh start)
        for (int t = 0; t < 2; ++t) {
            std::vector<double> data = {t * 100.0, t * 200.0};
            writer.writeToDataset(dset, H5T_NATIVE_DOUBLE, t, data.data(), {1, 2}, 0, 0, false);
        }
        writer.closeDataset(dset);
    }

    // Should have only 2 timesteps (not 5), with new values
    hid_t file = H5Fopen((filename + ".h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(file >= 0);
    hid_t read_dset = H5Dopen(file, "data", H5P_DEFAULT);
    hid_t space = H5Dget_space(read_dset);
    std::vector<hsize_t> read_dims(2);
    H5Sget_simple_extent_dims(space, read_dims.data(), NULL);
    CHECK(read_dims[0] == 2); // Only 2 timesteps (fresh file)
    CHECK(read_dims[1] == 2);

    // Verify new data values (multiplied by 100)
    std::vector<double> all_data(2 * 2);
    H5Dread(read_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, all_data.data());
    for (int t = 0; t < 2; ++t) {
        CHECK(all_data[t * 2 + 0] == doctest::Approx(t * 100.0));
        CHECK(all_data[t * 2 + 1] == doctest::Approx(t * 200.0));
    }

    H5Sclose(space);
    H5Dclose(read_dset);
    H5Fclose(file);

    // Cleanup
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::filesystem::remove("test_no_checkpoint.h5");
    }
    MPI_Barrier(MPI_COMM_WORLD); // wait for rank 0 to finish cleanup
}
#else
TEST_CASE("HDF5Writer - no HDF5 support") {
    CHECK_THROWS_AS(tndm::HDF5Writer("any", MPI_COMM_WORLD), std::runtime_error);
}
#endif
