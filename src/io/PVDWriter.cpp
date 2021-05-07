#include "PVDWriter.h"
#include "io/Endianness.h"
#include "tinyxml2.h"
#include <filesystem>

namespace tndm {
PVDWriter::PVDWriter(std::string_view baseName) : base_(baseName) {
    auto root = doc_.NewElement("VTKFile");
    doc_.InsertFirstChild(root);
    root->SetAttribute("type", "Collection");
    root->SetAttribute("version", "0.1");
    if (isBigEndian()) {
        root->SetAttribute("byte_order", "BigEndian");
    } else {
        root->SetAttribute("byte_order", "LittleEndian");
    }
    collection_ = root->InsertNewChildElement("Collection");
}

void PVDWriter::addTimestep(double time, std::string_view fileName) {
    auto relpath = std::filesystem::relative(
        std::filesystem::path(fileName),
        base_.has_parent_path() ? base_.parent_path() : std::filesystem::current_path());
    auto d = collection_->InsertNewChildElement("DataSet");
    d->SetAttribute("timestep", time);
    d->SetAttribute("group", "");
    d->SetAttribute("part", 0);
    d->SetAttribute("file", relpath.string().c_str());
}

bool PVDWriter::write() {
    std::string fileName = base_.string() + ".pvd";
    auto success = doc_.SaveFile(fileName.c_str());
    return success == tinyxml2::XML_SUCCESS;
}

} // namespace tndm
