#include "H5Parser.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdio.h>

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
    if (name == "/boundary") {
        hsize_t dims[1];
        H5Sget_simple_extent_dims(dataspace, dims, nullptr);
        data.resize(dims[0]);

        if (H5Dread(dataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
            H5Sclose(dataspace);
            H5Dclose(dataset);
        }
    } else if (name == "/connect") {
        hsize_t dims[2];
        H5Sget_simple_extent_dims(dataspace, dims, nullptr);
        data.resize(dims[0] * dims[1]);

        if (H5Dread(dataset, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
            H5Sclose(dataspace);
            H5Dclose(dataset);
        }
    } else if (name == "/geometry") {
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

    std::size_t numVertices = nodeData.size() / 3;
    builder->setNumVertices(numVertices);

    for (std::size_t i = 0; i < numVertices; ++i) {
        std::array<double, 3> x = {nodeData[i * 3], nodeData[i * 3 + 1], nodeData[i * 3 + 2]};
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
        higherOrderElements.push_back(nodes);
    }
    return true;
}

// Function to check if a sorted version of faceNodes exists in lowerOrderElements
bool isAlreadyPresent(const std::vector<std::array<long, 3>>& lowerOrderElements,
                      const std::array<long, 3>& faceNodes) {
    std::array<long, 3> sortedFaceNodes = faceNodes; // Copy to avoid modifying original
    std::sort(sortedFaceNodes.begin(), sortedFaceNodes.end());

    return std::any_of(lowerOrderElements.begin(), lowerOrderElements.end(),
                       [&](const std::array<long, 3>& elem) {
                           std::array<long, 3> sortedElem = elem;
                           std::sort(sortedElem.begin(), sortedElem.end());
                           return sortedElem == sortedFaceNodes;
                       });
}

bool H5Parser::retrieveLowerOrderElements(hid_t file) {
    auto& encodedBoundaryData = boundaryData;
    std::vector<std::uint8_t> decodedBoundary;

    const std::array<std::array<int, 3>, 4> sVertSeissol = {
        {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}}};
    int x = 0;
    for (size_t i = 0; i < higherOrderElements.size(); ++i) {
        const auto& elementNodes = higherOrderElements[i];
        uint32_t boundaryCondition = encodedBoundaryData[i];

        for (int face = 0; face < 4; ++face) {
            uint8_t faceTag = (boundaryCondition >> (8 * face)) & 0xFF;

            if (faceTag > 0) {
                x++;

                const auto& indices = sVertSeissol[face];
                std::array<long, 3> faceNodes = {elementNodes[indices[0]], elementNodes[indices[1]],
                                                 elementNodes[indices[2]]};

                // Check if a sorted version exists
                if (!isAlreadyPresent(lowerOrderElements, faceNodes)) {
                    lowerOrderElements.push_back(faceNodes); // Store as is, without sorting
                    boundary.push_back(faceTag);
                }
            }
        }
    }
    std::cout << "LOWER ORDER ELEMENT SIZE " << lowerOrderElements.size() << std::endl;
    std::cout << " COUNT of face tags " << x << std::endl;
    return true;
}

bool H5Parser::addAllElements(std::string const& fileName) {
    for (size_t i = 0; i < lowerOrderElements.size(); ++i) {
        long type = 2;
        builder->addElement(type, long(boundary[i]), lowerOrderElements[i].data(),
                            NumNodes[type - 1]);
    }

    for (auto& higherOrderElems : higherOrderElements) {
        long type = 4;
        builder->addElement(type, 0, higherOrderElems.data(), NumNodes[type - 1]);
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
    retrieveLowerOrderElements(file);
    addAllElements(fileName);

    H5Fclose(file);
    return true;
}

} // namespace tndm
