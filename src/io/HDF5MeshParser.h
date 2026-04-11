#ifndef HDF5MESHPARSER
#define HDF5MESHPARSER

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

class HDF5MeshParser : public MeshParser {
private:
    MeshBuilder* builder_;
    std::string errorMsg_;
    std::vector<std::array<long, 4>> higherDimensionalElements_;
    std::vector<std::array<long, 3>> lowerDimensionalElements_;
    std::vector<uint32_t> groupTags_;
    std::vector<uint32_t> boundaryData_;
    std::vector<uint8_t> boundary_;

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
    HDF5MeshParser(MeshBuilder* builder) : builder_(builder) {}

    bool parseFile(std::string const& fileName) override;
    std::string_view getErrorMessage() const override { return errorMsg_; }

    const std::vector<std::array<long, 4>>& getHigherDimensionalElements() const {
        return higherDimensionalElements_;
    }
    const std::vector<std::array<long, 3>>& getLowerDimensionalElements() const {
        return lowerDimensionalElements_;
    }
    const std::vector<uint32_t>& getBoundaryData() const { return boundaryData_; }
    const std::vector<uint8_t>& getBoundary() const { return boundary_; }
};

} // namespace tndm

#endif // HDF5MESHPARSER_H
