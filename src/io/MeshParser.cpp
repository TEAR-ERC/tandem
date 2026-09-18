#include "MeshParser.h"

#include "GMSHParser.h"
#include "H5Parser.h"

#include <filesystem>
#include <utility>

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

template <std::size_t D>
std::pair<std::unique_ptr<MeshParser>, std::string>
MeshParser::createWithValidation(std::string const& fileName, MeshBuilder* builder) {
    if (isH5Format(fileName)) {
#ifdef ENABLE_HDF5
        if constexpr (D != 3) {
            return {nullptr, "H5 mesh format is only supported for 3D problems."};
        }
#else
        return {nullptr, "HDF5 mesh support is not enabled."};
#endif
    }
    auto parser = create(fileName, builder);
    if (!parser) {
        return {nullptr, "Unsupported mesh file format: " + fileName};
    }
    return {std::move(parser), ""};
}

// Explicit template instantiations
template std::pair<std::unique_ptr<MeshParser>, std::string>
MeshParser::createWithValidation<2>(std::string const&, MeshBuilder*);
template std::pair<std::unique_ptr<MeshParser>, std::string>
MeshParser::createWithValidation<3>(std::string const&, MeshBuilder*);

} // namespace tndm
