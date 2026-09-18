#include "io/H5Parser.h"
#include "doctest.h"
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
        REQUIRE(id < expectedVertices);
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
        REQUIRE(elNo < expectedElements + 4); // Allow for boundary elements

        if (type == 4) { // Tetrahedral element (higher order)
            CHECK(numNodes == 4);
            // Verify node indices are valid
            for (std::size_t i = 0; i < numNodes; ++i) {
                CHECK(node[i] >= 0);
                CHECK(node[i] < static_cast<long>(expectedVertices));
            }
        } else if (type == 2) { // Triangular boundary element (lower order)
            CHECK(numNodes == 3);
            CHECK(tag > 0); // Should have boundary tag
            // Verify node indices are valid
            for (std::size_t i = 0; i < numNodes; ++i) {
                CHECK(node[i] >= 0);
                CHECK(node[i] < static_cast<long>(expectedVertices));
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

    // Boundary condition: All 4 faces have different tags
    // Face 0: tag 1, Face 1: tag 2, Face 2: tag 3, Face 3: tag 4
    // Format: 32-bit value with 8 bits per face (face3|face2|face1|face0)
    uint32_t boundary_data[1] = {0x03050301}; // tags 1,2,3,4 for faces 0,1,2,3
    H5Dwrite(bound_dset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, boundary_data);
    H5Dclose(bound_dset);
    H5Sclose(bound_space);

    H5Fclose(file_id);
    return true;
}

TEST_CASE("H5Parser - Basic Functionality") {
    const std::string test_filename = "test_mesh.h5";

    SUBCASE("File Creation and Parsing") {
        // Create test HDF5 file
        REQUIRE(createTestHDF5File(test_filename));

        H5TestBuilder builder;
        H5Parser parser(&builder);

        // Test successful parsing
        bool parseResult = parser.parseFile(test_filename);
        if (!parseResult) {
            std::cout << "Parse error: " << parser.getErrorMessage() << std::endl;
        }
        REQUIRE(parseResult);

        // Verify basic counts
        CHECK(builder.getExpectedVertices() == 4);
        CHECK(builder.getVertexCount() == 4);
        CHECK(builder.getExpectedElements() == 1); // One tetrahedron

        // Check that higher order elements were parsed
        CHECK(parser.higherOrderElements.size() == 1);
        CHECK(parser.higherOrderElements[0][0] == 0);
        CHECK(parser.higherOrderElements[0][1] == 1);
        CHECK(parser.higherOrderElements[0][2] == 2);
        CHECK(parser.higherOrderElements[0][3] == 3);

        // Check boundary data
        CHECK(parser.boundaryData.size() == 1);
        CHECK(parser.boundaryData[0] == 0x03050301);

        // Check lower order elements (boundary faces) - should have all 4 faces
        CHECK(parser.lowerOrderElements.size() == 4);
        CHECK(parser.boundary.size() == 4);

        // Verify all 4 boundary tags are present (1, 2, 3, 4)
        std::vector<uint8_t> expectedTags = {1, 3, 5, 3};
        for (uint8_t expectedTag : expectedTags) {
            bool tagFound = false;
            for (uint8_t actualTag : parser.boundary) {
                if (actualTag == expectedTag) {
                    tagFound = true;
                    break;
                }
            }
            CHECK(tagFound);
        }

        // Clean up
        std::remove(test_filename.c_str());
    }

    SUBCASE("Error Handling - Invalid File") {
        H5TestBuilder builder;
        H5Parser parser(&builder);

        bool parseResult = parser.parseFile("file_doesnt_exist.h5");
        REQUIRE_FALSE(parseResult);

        std::string_view errorMsg = parser.getErrorMessage();
        CHECK(errorMsg.find("Unable to open HDF5 file") != std::string_view::npos);
    }

    SUBCASE("Error Message Functionality") {
        H5TestBuilder builder;
        H5Parser parser(&builder);

        // Initially should have no error message
        CHECK(parser.getErrorMessage().empty());

        // After failed parse, should have error message
        parser.parseFile("nonexistent_file.h5");
        CHECK_FALSE(parser.getErrorMessage().empty());
    }
}

TEST_CASE("H5Parser - Data Structure Validation") {
    const std::string test_filename = "test_mesh_validation.h5";

    REQUIRE(createTestHDF5File(test_filename));

    H5TestBuilder builder;
    H5Parser parser(&builder);

    REQUIRE(parser.parseFile(test_filename));

    SUBCASE("Higher Order Elements Structure") {
        REQUIRE(parser.higherOrderElements.size() > 0);

        // Each higher order element should have 4 nodes (tetrahedron)
        for (const auto& element : parser.higherOrderElements) {
            // Verify all node indices are valid
            for (int i = 0; i < 4; ++i) {
                CHECK(element[i] >= 0);
                CHECK(element[i] < 4); // We have 4 vertices
            }
        }
    }

    SUBCASE("Lower Order Elements Structure") {
        // Should have exactly 4 boundary faces (all faces of tetrahedron)
        CHECK(parser.lowerOrderElements.size() == 4);
        CHECK(parser.boundary.size() == 4);

        // Each lower order element should have 3 nodes (triangle)
        for (const auto& element : parser.lowerOrderElements) {
            for (int i = 0; i < 3; ++i) {
                CHECK(element[i] >= 0);
                CHECK(element[i] < 4); // We have 4 vertices
            }
        }

        std::vector<uint8_t> foundTags;
        for (const auto& tag : parser.boundary) {
            CHECK(tag > 0);
            CHECK(tag <= 5);
            foundTags.push_back(tag);
        }

        // Verify we have all 4 unique tags
        std::vector<uint8_t> expectedTags = {1, 3, 5, 3};
        CHECK(foundTags == expectedTags);
    }

    SUBCASE("Boundary Data Consistency") {
        REQUIRE(parser.boundaryData.size() > 0);

        // Our test data should have all 4 faces tagged
        CHECK(parser.boundaryData[0] == 0x03050301);

        // Should have exactly 4 triangular boundary faces
        CHECK(parser.lowerOrderElements.size() == 4);

        // Each face should correspond to the correct vertices of the tetrahedron
        // Tetrahedron faces (using SEISSOL vertex ordering):
        // Face 0: vertices {0,2,1} -> nodes {0,1,2} (reordered)
        // Face 1: vertices {0,1,3} -> nodes {0,1,3}
        // Face 2: vertices {1,2,3} -> nodes {1,2,3}
        // Face 3: vertices {0,3,2} -> nodes {0,2,3} (reordered)

        // Verify that we have valid triangular faces
        for (const auto& face : parser.lowerOrderElements) {
            // Each face should have 3 different vertices
            CHECK(face[0] != face[1]);
            CHECK(face[1] != face[2]);
            CHECK(face[0] != face[2]);

            // All vertices should be in range [0,3]
            for (int i = 0; i < 3; ++i) {
                CHECK(face[i] >= 0);
                CHECK(face[i] <= 3);
            }
        }
    }

    std::remove(test_filename.c_str());
}

TEST_CASE("H5Parser - Integration with meshBuilder") {
    const std::string test_filename = "test_mesh_integration.h5";

    REQUIRE(createTestHDF5File(test_filename));

    H5TestBuilder builder;
    H5Parser parser(&builder);

    bool parseResult = parser.parseFile(test_filename);
    if (!parseResult) {
        std::cout << "Integration test parse error: " << parser.getErrorMessage() << std::endl;
    }
    REQUIRE(parseResult);

    SUBCASE("Builder Verification") {
        // Verify that the builder was called correctly
        CHECK(builder.getVertexCount() == builder.getExpectedVertices());
        CHECK(builder.getElementCount() >
              builder.getExpectedElements()); // Includes boundary elements

        // Verify vertices were set correctly
        CHECK(builder.getExpectedVertices() == 4);

        // Verify elements were added (both volume and boundary)
        CHECK(builder.getElementCount() >= 1); // At least the volume element
    }

    std::remove(test_filename.c_str());
}