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
    template <typename T> T logErrorAnnotated(std::string_view msg);

#ifdef ENABLE_HDF5
    template <typename T, std::size_t Rank>
    bool readDataset(hid_t file, std::string_view name, std::vector<T>& data);
    bool parseNodes(hid_t file);
    bool parseElements(hid_t file);
    bool readBoundaryData(hid_t file);
    bool retrieveLowerDimensionalElements(hid_t file);
    bool addAllElements(std::string const& fileName);
#endif

public:
    H5Parser(MeshBuilder* builder) : builder(builder) {}

    bool parseFile(std::string const& fileName) override;
    std::string_view getErrorMessage() const override { return errorMsg; }
};

} // namespace tndm

#endif // H5PARSER_H
