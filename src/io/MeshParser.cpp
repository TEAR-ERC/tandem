#include "MeshParser.h"

#include "GMSHParser.h"
#include "H5Parser.h"

#include <filesystem>

namespace tndm {

bool MeshParser::isGMSHFormat(std::string const& fileName) {
    return std::filesystem::path(fileName).extension() == ".msh";
}

bool MeshParser::isH5Format(std::string const& fileName) {
    return std::filesystem::path(fileName).extension() == ".h5";
}

std::unique_ptr<MeshParser> MeshParser::create(std::string const& fileName, MeshBuilder* builder) {
    if (isGMSHFormat(fileName)) {
        return std::make_unique<GMSHParser>(builder);
    }
    if (isH5Format(fileName)) {
#ifdef ENABLE_HDF5
        return std::make_unique<H5Parser>(builder);
#else
        return nullptr;
#endif
    }
    return nullptr;
}

} // namespace tndm
