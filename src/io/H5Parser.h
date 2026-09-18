#ifndef H5PARSER_H
#define H5PARSER_H

#include "MeshParser.h"
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#ifdef ENABLE_HDF5
#include <hdf5.h>
#endif

namespace tndm {

class H5Parser : public MeshParser {
private:
    MeshBuilder* builder;
    std::string errorMsg;
    std::vector<std::array<long, 4>> higherDimensionalElements;
    std::vector<std::array<long, 3>> lowerDimensionalElements;
    std::vector<uint32_t> groupTags;
    std::vector<uint32_t> boundaryData;
    std::vector<uint8_t> boundary;

    template <typename T> T logError(std::string_view msg);

#ifdef ENABLE_HDF5
    template <typename T> bool readDataset(hid_t file, std::string_view name, std::vector<T>& data);
    bool parseNodes(hid_t file);
    bool parseElements(hid_t file);
    bool readBoundaryData(hid_t file);
    bool retrieveLowerDimensionalElements();
    bool addAllElements();
#endif

public:
    H5Parser(MeshBuilder* builder) : builder(builder) {}

    bool parseFile(std::string const& fileName) override;
    std::string_view getErrorMessage() const override { return errorMsg; }

    const std::vector<std::array<long, 4>>& getHigherDimensionalElements() const {
        return higherDimensionalElements;
    }
    const std::vector<std::array<long, 3>>& getLowerDimensionalElements() const {
        return lowerDimensionalElements;
    }
    const std::vector<uint32_t>& getBoundaryData() const { return boundaryData; }
    const std::vector<uint8_t>& getBoundary() const { return boundary; }
};

} // namespace tndm

#endif // H5PARSER_H
