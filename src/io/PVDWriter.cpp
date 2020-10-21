#include "PVDWriter.h"
#include "io/Endianness.h"
#include "tinyxml2.h"

namespace tndm {
PVDWriter::PVDWriter() {
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

void PVDWriter::addTimestep(double time, std::string const& fileName) {
    auto d = collection_->InsertNewChildElement("DataSet");
    d->SetAttribute("timestep", time);
    d->SetAttribute("group", "");
    d->SetAttribute("part", 0);
    d->SetAttribute("file", fileName.c_str());
}

bool PVDWriter::write(std::string const& baseName) {
    std::string fileName = baseName + ".pvd";
    auto success = doc_.SaveFile(fileName.c_str());
    return success == tinyxml2::XML_SUCCESS;
}

} // namespace tndm
