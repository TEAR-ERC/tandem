#include "io/H5Parser.h"
#include "doctest.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string_view>

using namespace tndm;

class H5TestBuilder : public meshBuilder {
private:
    std::size_t elNo = 0;
    std::size_t vertexCount = 0;
    std::size_t expectedVertices = 0;
    std::size_t expectedElements = 0;

public:
    void setNumVertices(std::size_t numVertices) override {
        expectedVertices = numVertices;
        CHECK(numVertices > 0);
    }

    void setVertex(long id, std::array<double, 3> const& x) override {
        // Test case: simple tetrahedron mesh with 4 vertices
        std::array<std::array<double, 3>, 4> reference = {
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
        REQUIRE(id < (long)expectedVertices);
        if (expectedVertices == 4 && id < 4) {
            for (std::size_t i = 0; i < 3; ++i) {
                CHECK(x[i] == doctest::Approx(reference[id][i]));
            }
        }
        vertexCount++;
    }

    void setNumElements(std::size_t numElements) override {
        expectedElements = numElements;
        CHECK(numElements > 0);
    }

    void addElement(long type, long tag, long* node, std::size_t numNodes) override {
        if (type == 4) { // Tetrahedral element (higher Dimensional)
            CHECK(numNodes == 4);
            for (std::size_t i = 0; i < numNodes; ++i) {
                CHECK(node[i] >= 0);
                CHECK(node[i] < (long)expectedVertices);
            }
        } else if (type == 2) { // Triangular boundary element (lower Dimensional)
            CHECK(numNodes == 3);
            CHECK(tag > 0); // Should have boundary tag
            // Verify node indices are valid
            for (std::size_t i = 0; i < numNodes; ++i) {
                CHECK(node[i] >= 0);
                CHECK(node[i] < (long)expectedVertices);
            }
        }
        ++elNo;
    }

    // Test getters
    std::size_t getVertexCount() const { return vertexCount; }
    std::size_t getElementCount() const { return elNo; }
    std::size_t getExpectedVertices() const { return expectedVertices; }
    std::size_t getExpectedElements() const { return expectedElements; }
};

// Helper function to create a simple test HDF5 file
bool createTestHDF5File(const std::string& filename) {
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
        return false;

    // Create geometry dataset (4 vertices x 3 coordinates)
    hsize_t geom_dims[2] = {4, 3};
    hid_t geom_space = H5Screate_simple(2, geom_dims, NULL);
    hid_t geom_dset = H5Dcreate2(file_id, "/geometry", H5T_NATIVE_DOUBLE, geom_space, H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);

    double geometry_data[12] = {
        0.0, 0.0, 0.0, // vertex 0
        1.0, 0.0, 0.0, // vertex 1
        0.0, 1.0, 0.0, // vertex 2
        0.0, 0.0, 1.0  // vertex 3
    };
    H5Dwrite(geom_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, geometry_data);
    H5Dclose(geom_dset);
    H5Sclose(geom_space);

    // Create connectivity dataset (1 tetrahedron x 4 nodes)
    hsize_t conn_dims[2] = {1, 4};
    hid_t conn_space = H5Screate_simple(2, conn_dims, NULL);
    hid_t conn_dset = H5Dcreate2(file_id, "/connect", H5T_NATIVE_LONG, conn_space, H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);

    long connectivity_data[4] = {0, 1, 2, 3}; // Single tetrahedron
    H5Dwrite(conn_dset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, connectivity_data);
    H5Dclose(conn_dset);
    H5Sclose(conn_space);

    // Create boundary dataset (1 element with boundary conditions)
    hsize_t bound_dims[1] = {1};
    hid_t bound_space = H5Screate_simple(1, bound_dims, NULL);
    hid_t bound_dset = H5Dcreate2(file_id, "/boundary", H5T_NATIVE_UINT32, bound_space, H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);

    // Boundary condition encoded as 0x03050301 -> faces (0..3) = {1,3,5,3}
    uint32_t boundary_data[1] = {0x03050301u};
    H5Dwrite(bound_dset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, boundary_data);
    H5Dclose(bound_dset);
    H5Sclose(bound_space);

    H5Fclose(file_id);
    return true;
}

// Two tetrahedra sharing face {0,1,2}; that face is tagged from both sides -> must appear once
bool createTwoTetHDF5File(const std::string& filename) {
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        return false;
    }
    hsize_t geom_dims[2] = {5, 3};
    hid_t geom_space = H5Screate_simple(2, geom_dims, NULL);
    hid_t geom_dset = H5Dcreate2(file_id, "/geometry", H5T_NATIVE_DOUBLE, geom_space, H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);
    double geometry_data[15] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    H5Dwrite(geom_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, geometry_data);
    H5Dclose(geom_dset);
    H5Sclose(geom_space);

    hsize_t conn_dims[2] = {2, 4};
    hid_t conn_space = H5Screate_simple(2, conn_dims, NULL);
    hid_t conn_dset = H5Dcreate2(file_id, "/connect", H5T_NATIVE_LONG, conn_space, H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);
    long connectivity_data[8] = {0, 1, 2, 3, 0, 1, 2, 4};
    H5Dwrite(conn_dset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, connectivity_data);
    H5Dclose(conn_dset);
    H5Sclose(conn_space);

    hsize_t bound_dims[1] = {2};
    hid_t bound_space = H5Screate_simple(1, bound_dims, NULL);
    hid_t bound_dset = H5Dcreate2(file_id, "/boundary", H5T_NATIVE_UINT32, bound_space, H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
    // tet0: face0=9,face1=2,face2=3,face3=4 -> 0x04030209
    // tet1: face0=9,face1=5,face2=6,face3=7 -> 0x07060509
    uint32_t boundary_data[2] = {0x04030209u, 0x07060509u};
    H5Dwrite(bound_dset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, boundary_data);
    H5Dclose(bound_dset);
    H5Sclose(bound_space);

    H5Fclose(file_id);
    return true;
}

// Single tetrahedron where only 2 of 4 faces are tagged
bool createPartialBoundaryHDF5File(const std::string& filename) {
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        return false;
    }
    hsize_t geom_dims[2] = {4, 3};
    hid_t geom_space = H5Screate_simple(2, geom_dims, NULL);
    hid_t geom_dset = H5Dcreate2(file_id, "/geometry", H5T_NATIVE_DOUBLE, geom_space, H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);
    double geometry_data[12] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
    H5Dwrite(geom_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, geometry_data);
    H5Dclose(geom_dset);
    H5Sclose(geom_space);

    hsize_t conn_dims[2] = {1, 4};
    hid_t conn_space = H5Screate_simple(2, conn_dims, NULL);
    hid_t conn_dset = H5Dcreate2(file_id, "/connect", H5T_NATIVE_LONG, conn_space, H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);
    long connectivity_data[4] = {0, 1, 2, 3};
    H5Dwrite(conn_dset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, connectivity_data);
    H5Dclose(conn_dset);
    H5Sclose(conn_space);

    hsize_t bound_dims[1] = {1};
    hid_t bound_space = H5Screate_simple(1, bound_dims, NULL);
    hid_t bound_dset = H5Dcreate2(file_id, "/boundary", H5T_NATIVE_UINT32, bound_space, H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
    // Only face0 (tag=1) and face2 (tag=5) tagged; faces 1 and 3 are zero
    // 0x00050001 -> face0=1, face1=0, face2=5, face3=0
    uint32_t boundary_data[1] = {0x00050001u};
    H5Dwrite(bound_dset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, boundary_data);
    H5Dclose(bound_dset);
    H5Sclose(bound_space);

    H5Fclose(file_id);
    return true;
}

// Test cases for H5Parser
TEST_CASE("H5Parser - Single tetrahedron, all faces tagged") {
    const std::string fname = "test_single_tet.h5";
    REQUIRE(createTestHDF5File(fname));

    H5TestBuilder builder;
    H5Parser parser(&builder);
    REQUIRE(parser.parseFile(fname));

    CHECK(builder.getExpectedVertices() == 4);
    CHECK(builder.getVertexCount() == 4);
    CHECK(builder.getExpectedElements() == 1);

    CHECK(parser.higherDimensionalElements.size() == 1);
    CHECK(parser.higherDimensionalElements[0] == std::array<long, 4>{0, 1, 2, 3});

    CHECK(parser.boundaryData[0] == 0x03050301u);

    // All 4 faces tagged -> 4 boundary triangles
    CHECK(parser.lowerDimensionalElements.size() == 4);
    CHECK(parser.boundary.size() == 4);

    // Tags must match the byte-by-byte decoding order
    CHECK(parser.boundary == std::vector<uint8_t>{1, 3, 5, 3});

    // Every face must have 3 distinct valid node indices
    for (const auto& face : parser.lowerDimensionalElements) {
        CHECK(face[0] != face[1]);
        CHECK(face[1] != face[2]);
        CHECK(face[0] != face[2]);
        for (int i = 0; i < 3; ++i) {
            CHECK(face[i] >= 0);
            CHECK(face[i] <= 3);
        }
    }

    std::remove(fname.c_str());
}

TEST_CASE("H5Parser - Duplicate face deduplication (two tets sharing a face)") {
    const std::string fname = "test_two_tet.h5";
    REQUIRE(createTwoTetHDF5File(fname));

    H5TestBuilder builder;
    H5Parser parser(&builder);
    REQUIRE(parser.parseFile(fname));

    // 4 faces per tet, 1 shared face tagged on both sides -> 7 unique boundary faces
    CHECK(parser.lowerDimensionalElements.size() == 7);
    CHECK(parser.boundary.size() == 7);

    // Shared face {0,1,2} must appear exactly once
    std::array<long, 3> sharedSorted = {0, 1, 2};
    int count = 0;
    for (auto& face : parser.lowerDimensionalElements) {
        std::array<long, 3> s = face;
        std::sort(s.begin(), s.end());
        if (s == sharedSorted)
            ++count;
    }
    CHECK(count == 1);

    std::remove(fname.c_str());
}

TEST_CASE("H5Parser - Partial boundary (only some faces tagged)") {
    const std::string fname = "test_partial_boundary.h5";
    REQUIRE(createPartialBoundaryHDF5File(fname));

    H5TestBuilder builder;
    H5Parser parser(&builder);
    REQUIRE(parser.parseFile(fname));

    // Only 2 of the 4 faces are tagged
    CHECK(parser.lowerDimensionalElements.size() == 2);
    CHECK(parser.boundary.size() == 2);

    // Tags should be exactly the two non-zero ones, in face order
    CHECK(parser.boundary == std::vector<uint8_t>{1, 5});

    std::remove(fname.c_str());
}

TEST_CASE("H5Parser - Error handling") {
    SUBCASE("Nonexistent file returns false with message") {
        H5TestBuilder builder;
        H5Parser parser(&builder);

        REQUIRE_FALSE(parser.parseFile("does_not_exist.h5"));
        CHECK(parser.getErrorMessage().find("Unable to open HDF5 file") != std::string::npos);
    }

    SUBCASE("No error message before any parse attempt") {
        H5TestBuilder builder;
        H5Parser parser(&builder);
        CHECK(parser.getErrorMessage().empty());
    }
}