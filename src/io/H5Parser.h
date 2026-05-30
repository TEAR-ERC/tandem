#ifndef H5PARSER_20250120_H
#define H5PARSER_20250120_H

#include "/home/pkarki/hdf5/include/hdf5.h"
#include "meshParser.h"
#include </home/pkarki/hdf5/include/H5FDmpi.h>
#include </home/pkarki/hdf5/include/H5FDmpio.h>
#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tndm {

class H5Parser : public meshParser {
private:
    meshBuilder* builder;
    std::string errorMsg;

    template <typename T> T logError(std::string_view msg);
    template <typename T> T logErrorAnnotated(std::string_view msg);
    template <typename T> bool readDataset(hid_t file, const char* name, std::vector<T>& data);
    bool parseNodes(hid_t file);
    bool parseElements(hid_t file);
    bool parseBoundary(hid_t file);
    bool retrieveLowerOrderElements(hid_t file);

public:
    H5Parser(meshBuilder* builder) : builder(builder) {}

    bool parseFile(std::string const& fileName) override;
    bool addAllElements(std::string const& fileName);

    std::string_view getErrorMessage() const override { return errorMsg; }

    std::vector<std::array<long, 4>> higherOrderElements;
    std::vector<std::array<long, 3>> lowerOrderElements;
    std::vector<std::uint32_t> boundaryData;
    std::vector<std::uint8_t> boundary;
};

} // namespace tndm

#endif // H5PARSER_20250120_H
