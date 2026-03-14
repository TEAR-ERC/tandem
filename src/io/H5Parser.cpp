#include "H5Parser.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdio.h>
#include <unordered_set>

namespace tndm {

template <typename T> T H5Parser::logError(std::string_view msg) {
    errorMsg += "H5 parser error:\n\t";
    errorMsg += msg;
    errorMsg += '\n';
    return {};
}

template <typename T> T H5Parser::logErrorAnnotated(std::string_view msg) {
    errorMsg += "H5 parser error:\n\t";
    errorMsg += msg;
    errorMsg += '\n';
    return {};
}

template <typename T>
bool H5Parser::readDataset(hid_t file, const char* name, std::vector<T>& data) {

    hid_t dataset = H5Dopen(file, name, H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(dataset);
    if (dataspace < 0) {
        H5Dclose(dataset);
    }
    if (name == "/boundary") { // Boundary conditions in 32 bit encoding
        hsize_t dims[1];
        H5Sget_simple_extent_dims(dataspace, dims, nullptr);
        data.resize(dims[0]);

        if (H5Dread(dataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
            H5Sclose(dataspace);
            H5Dclose(dataset);
        }
    } else if (name == "/connect") { // Nodes associated with each higher Dimensional
                                     // element(higherDimensionalElements x 4 for 4 node element)
        hsize_t dims[2];
        H5Sget_simple_extent_dims(dataspace, dims, nullptr);
        data.resize(dims[0] * dims[1]);

        if (H5Dread(dataset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
            H5Sclose(dataspace);
            H5Dclose(dataset);
        }
    } else if (name ==
               "/geometry") { // Coordinates of the nodes in the mesh (number of Nodes x Dimensions)
        hsize_t dims[2];
        H5Sget_simple_extent_dims(dataspace, dims, nullptr);
        data.resize(dims[0] * dims[1]);

        if (H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
            H5Sclose(dataspace);
            H5Dclose(dataset);
        }
    } else {
        std::cout << "Invalid dataset name" << std::endl;
    }

    H5Sclose(dataspace);
    H5Dclose(dataset);
    return true;
}

bool H5Parser::parseNodes(hid_t file) {
    std::vector<double> nodeData;
    if (!readDataset<double>(file, "/geometry", nodeData)) {
        return logErrorAnnotated<bool>("Failed to parse nodes");
    }
    constexpr std::size_t DomainDimension = 3;
    std::size_t numVertices = nodeData.size() / DomainDimension;
    builder->setNumVertices(numVertices);

    for (std::size_t i = 0; i < numVertices; ++i) {
        std::array<double, DomainDimension> x = {nodeData[i * DomainDimension],
                                                 nodeData[i * DomainDimension + 1],
                                                 nodeData[i * DomainDimension + 2]};
        builder->setVertex(i, x);
    }

    return true;
}

bool H5Parser::parseBoundary(hid_t file) {
    if (!readDataset<uint32_t>(file, "/boundary", boundaryData)) {
        return logErrorAnnotated<bool>("Failed to parse nodes");
    }
    return true;
}

bool H5Parser::parseElements(hid_t file) {
    std::vector<long> elementData;
    if (!readDataset<long>(file, "/connect", elementData)) {
        return logErrorAnnotated<bool>("Failed to parse elements");
    }

    const size_t elementSides = 4;
    std::size_t numElements = elementData.size() / elementSides;
    builder->setNumElements(numElements);

    for (std::size_t i = 0; i < numElements; ++i) {
        std::array<long, 4> nodes = {
            elementData[i * elementSides], elementData[i * elementSides + 1],
            elementData[i * elementSides + 2], elementData[i * elementSides + 3]};
        higherDimensionalElements.push_back(nodes);
    }
    return true;
}

// Retrieve faces from tets and deduplicate them based on tags
bool H5Parser::retrieveLowerDimensionalElements(hid_t file) {

    // Each tet has 4 faces. This table maps each face index (0-3) to the
    // 3 local vertex indices that make up that face, in SeisSol ordering.
    constexpr std::size_t DomainDimension = 3;
    const std::array<std::array<int, DomainDimension>, 4> sVertSeissol = {
        {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}}};

    // FaceKey holds a sorted (canonical) face so that the same triangle
    // is recognised regardless of vertex winding order.
    struct FaceKey {
        std::array<long, FaceSize> v;
        bool operator==(FaceKey const& o) const noexcept { return v == o.v; }
    };
    struct FaceHash {
        std::size_t operator()(FaceKey const& k) const noexcept {
            std::size_t h = std::hash<long>{}(k.v[0]);
            h = (h * 1315423911u) ^ std::hash<long>{}(k.v[1]);
            h = (h * 1315423911u) ^ std::hash<long>{}(k.v[2]);
            return h;
        }
    };

    // Tracks which faces have already been added, to avoid duplicates.
    std::unordered_set<FaceKey, FaceHash> seen;
    seen.reserve(higherDimensionalElements.size() * 3);

    for (size_t i = 0; i < higherDimensionalElements.size(); ++i) {
        const auto& tetNodes = higherDimensionalElements[i];

        // The boundary condition for this tet is packed as four 8-bit tags,
        // one per face: bits [7:0]=face0, [15:8]=face1, [23:16]=face2, [31:24]=face3.
        const uint32_t boundaryCondition = boundaryData[i];

        for (int face = 0; face < 4; ++face) {

            // Extract the tag for this face. A tag of 0 means interior (skip).
            const uint8_t faceTag = (boundaryCondition >> (8 * face)) & 0xFF;
            if (faceTag == 0) {
                continue;
            }

            // Look up the 3 global node IDs for this face using the SeisSol table.
            const auto& localIndices = sVertSeissol[face];
            const std::array<long, FaceSize> faceNodes = {
                tetNodes[localIndices[0]], tetNodes[localIndices[1]], tetNodes[localIndices[2]]};

            // Sort the nodes to get a key for duplicate detection.
            // The original (unsorted) winding order is preserved in faceNodes.
            std::array<long, FaceSize> sortedNodes = faceNodes;
            std::sort(sortedNodes.begin(), sortedNodes.end());

            // Only add if this face hasn't been seen before.
            if (seen.insert(FaceKey{sortedNodes}).second) {
                lowerDimensionalElements.push_back(faceNodes);
                boundary.push_back(faceTag);
            }
        }
    }
    return true;
}

bool H5Parser::addAllElements(std::string const& fileName) {
    for (size_t i = 0; i < lowerDimensionalElements.size(); ++i) {
        // Assuming only 3 node triangular faces for lower Dimensional elements
        long type = 2;
        builder->addElement(type, long(boundary[i]), lowerDimensionalElements[i].data(),
                            NumNodes[type - 1]);
    }

    for (auto& higherDimensionalElems : higherDimensionalElements) {
        // Assuming only 4 node tetrahedral for higher Dimensional elements
        long type = 4;
        // Information not available in the HDF5 file from PUMGen  - but also not used by the
        // builder for higher Dimensional simplices
        long groupTage = 0;
        builder->addElement(type, 0, higherDimensionalElems.data(), NumNodes[type - 1]);
    }

    return true;
}

bool H5Parser::parseFile(std::string const& fileName) {
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    hid_t file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, plist_id);
    if (file < 0) {
        return logError<bool>("Unable to open HDF5 file: " + fileName);
    }
    parseElements(file);
    parseNodes(file);
    parseBoundary(file);
    retrieveLowerDimensionalElements(file);
    addAllElements(fileName);

    H5Fclose(file);
    return true;
}

} // namespace tndm
