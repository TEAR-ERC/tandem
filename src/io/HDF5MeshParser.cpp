#include "HDF5MeshParser.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

namespace tndm {

#ifdef ENABLE_HDF5

template <typename T> T HDF5MeshParser::logError(std::string_view msg) {
    errorMsg += "H5 parser error:\n\t";
    errorMsg += msg;
    errorMsg += '\n';
    return {};
}

template <typename T> hid_t nativeH5Type();
template <> hid_t nativeH5Type<double>() { return H5T_NATIVE_DOUBLE; }
template <> hid_t nativeH5Type<long>() { return H5T_NATIVE_LONG; }
template <> hid_t nativeH5Type<uint32_t>() { return H5T_NATIVE_UINT32; }

template <typename T>
bool HDF5MeshParser::readDataset(hid_t file, std::string_view name, std::vector<T>& data) {
    hid_t dataset = H5Dopen(file, std::string(name).c_str(), H5P_DEFAULT);
    if (dataset < 0) {
        return logError<bool>("Failed to open dataset: " + std::string(name));
    }

    hid_t dataspace = H5Dget_space(dataset);
    if (dataspace < 0) {
        H5Dclose(dataset);
        return logError<bool>("Failed to get dataset space: " + std::string(name));
    }

    hsize_t dims[2];
    int ndims = H5Sget_simple_extent_ndims(dataspace);
    if (ndims < 0) {
        H5Sclose(dataspace);
        H5Dclose(dataset);
        return logError<bool>("Unexpected dataset rank for: " + std::string(name));
    }
    if (H5Sget_simple_extent_dims(dataspace, dims, nullptr) < 0) {
        H5Sclose(dataspace);
        H5Dclose(dataset);
        return logError<bool>("Failed to read dataset dimensions: " + std::string(name));
    }

    std::size_t size = 1;
    for (int i = 0; i < ndims; ++i)
        size *= static_cast<std::size_t>(dims[i]);
    data.resize(size);

    if (H5Dread(dataset, nativeH5Type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
        H5Sclose(dataspace);
        H5Dclose(dataset);
        return logError<bool>("Failed to read dataset: " + std::string(name));
    }

    H5Sclose(dataspace);
    H5Dclose(dataset);
    return true;
}

bool HDF5MeshParser::parseNodes(hid_t file) {
    std::vector<double> nodeData;
    if (!readDataset<double>(file, "/geometry", nodeData)) {
        return logError<bool>("Failed to parse nodes");
    }
    constexpr std::size_t DOMAIN_DIMENSION = 3;
    std::size_t numVertices = nodeData.size() / DOMAIN_DIMENSION;
    builder->setNumVertices(numVertices);

    for (std::size_t i = 0; i < numVertices; ++i) {
        std::array<double, DOMAIN_DIMENSION> x = {nodeData[i * DOMAIN_DIMENSION],
                                                  nodeData[i * DOMAIN_DIMENSION + 1],
                                                  nodeData[i * DOMAIN_DIMENSION + 2]};
        // H5/PUMGen vertex ids are already 0-based, and the builder expects 0-based ids too.
        builder->setVertex(i, x);
    }
    return true;
}

bool HDF5MeshParser::readBoundaryData(hid_t file) {
    if (!readDataset<uint32_t>(file, "/boundary", boundaryData)) {
        return logError<bool>("Failed to parse boundary");
    }
    return true;
}

bool HDF5MeshParser::parseElements(hid_t file) {
    std::vector<long> elementData;
    if (!readDataset<long>(file, "/connect", elementData)) {
        return logError<bool>("Failed to parse elements");
    }
    if (!readDataset<uint32_t>(file, "/group", groupTags)) {
        // /group is optional; default all group tags to 0 if absent.
        errorMsg.clear();
        groupTags.assign(elementData.size() / 4, 0u);
    }

    constexpr std::size_t ELEMENT_SIDES = 4;
    std::size_t numElements = elementData.size() / ELEMENT_SIDES;

    for (std::size_t i = 0; i < numElements; ++i) {
        // PUMGen connectivity is already 0-based, which matches the builder's expectations.
        std::array<long, 4> nodes = {
            elementData[i * ELEMENT_SIDES], elementData[i * ELEMENT_SIDES + 1],
            elementData[i * ELEMENT_SIDES + 2], elementData[i * ELEMENT_SIDES + 3]};
        higherDimensionalElements.push_back(nodes);
    }
    return true;
}

// Retrieve faces from tets and deduplicate them based on tags
bool HDF5MeshParser::retrieveLowerDimensionalElements() {

    // Each tet has 4 faces. This table maps each face index (0-3) to the
    // 3 local vertex indices that make up that face, in SeisSol ordering.
    constexpr std::size_t DOMAIN_DIMENSION = 3;
    constexpr std::array<std::array<int, DOMAIN_DIMENSION>, 4> SVERT_SEISSOL = {
        {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}}};

    struct FaceKey {
        std::array<long, DOMAIN_DIMENSION> v;
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
            const auto& localIndices = SVERT_SEISSOL[face];
            const std::array<long, DOMAIN_DIMENSION> faceNodes = {
                tetNodes[localIndices[0]], tetNodes[localIndices[1]], tetNodes[localIndices[2]]};

            // Sort the nodes to get a key for duplicate detection.
            // The original (unsorted) winding order is preserved in faceNodes.
            std::array<long, DOMAIN_DIMENSION> sortedNodes = faceNodes;
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

bool HDF5MeshParser::addAllElements() {
    for (size_t i = 0; i < lowerDimensionalElements.size(); ++i) {
        // Assuming only 3 node triangular faces for lower Dimensional elements
        constexpr long TRIANGLE_TYPE = 2;
        builder->addElement(TRIANGLE_TYPE, long(boundary[i]), lowerDimensionalElements[i].data(),
                            NumNodes[TRIANGLE_TYPE - 1]);
    }
    for (size_t i = 0; i < higherDimensionalElements.size(); ++i) {
        // Assuming only 4 node tetrahedral for higher Dimensional elements
        constexpr long TET_TYPE = 4;
        // Information not available in the HDF5 file from PUMGen  - but also not used by the
        // builder for higher Dimensional simplices
        builder->addElement(TET_TYPE, static_cast<long>(groupTags[i]),
                            higherDimensionalElements[i].data(), NumNodes[TET_TYPE - 1]);
    }
    return true;
}

bool HDF5MeshParser::parseFile(std::string const& fileName) {
    errorMsg.clear();
    higherDimensionalElements.clear();
    lowerDimensionalElements.clear();
    groupTags.clear();
    boundaryData.clear();
    boundary.clear();

    hid_t file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        return logError<bool>("Unable to open HDF5 file: " + fileName);
    }

    bool ok = parseElements(file) && parseNodes(file) && readBoundaryData(file) &&
              (boundaryData.size() == higherDimensionalElements.size() ||
               logError<bool>("Boundary tag count does not match connectivity")) &&
              retrieveLowerDimensionalElements() && addAllElements();

    H5Fclose(file);
    return ok;
}

#else  // ENABLE_HDF5 not defined — stub definition so the linker is satisfied

bool HDF5MeshParser::parseFile(std::string const& fileName) {
    throw std::runtime_error("HDF5MeshParser::parseFile: tandem was built without HDF5 support. "
                             "Cannot parse '" +
                             fileName +
                             "'. "
                             "Reconfigure with -DENABLE_HDF5=ON.");
}
#endif // ENABLE_HDF5

} // namespace tndm
