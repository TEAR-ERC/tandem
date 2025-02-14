#include "PVDWriter.h"
#include "io/Endianness.h"
#include "tinyxml2.h"
#include <filesystem>
#include <iostream>
#include <vector>

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

void PVDWriter::load_from_file() {
    std::string fileName = base_.string() + ".pvd";
    doc_.Clear(); // Clear any previously loaded XML content
    auto result = doc_.LoadFile(fileName.c_str());
    if (result != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error("Error loading PVD file: " + fileName);
    }
    // Find the root element and the collection
    auto root = doc_.FirstChildElement("VTKFile");
    if (!root) {
        throw std::runtime_error("Missing root <VTKFile> element in " + fileName);
    }

    collection_ = root->FirstChildElement("Collection");
    if (!collection_) {
        throw std::runtime_error("Missing <Collection> element in " + fileName);
    }
}

void PVDWriter::truncate_after_restart(std::size_t lastTimestepIndex) {
    load_from_file();
    std::string fileName = base_.string() + ".pvd";

    if (!collection_) {
        throw std::runtime_error("Error: No Collection element found in " + fileName + "\n");
        return;
    }

    // Create a copy of the elements for safe iteration
    std::vector<tinyxml2::XMLElement*> datasets;
    for (tinyxml2::XMLElement* dataset = collection_->FirstChildElement("DataSet");
         dataset != nullptr; dataset = dataset->NextSiblingElement("DataSet")) {
        datasets.push_back(dataset);
    }

    std::size_t index = 0;
    for (auto* dataset : datasets) {
        if (index >= lastTimestepIndex) {
            collection_->DeleteChild(dataset);
        }
        index++;
    }

    if (index < lastTimestepIndex) {
        std::cerr << lastTimestepIndex << " datasets expected in " << fileName
                  << " when truncating (checkpoint restart), but read only " << index << "\n";
    }

    std::cout << "Done truncating " << fileName << " to " << lastTimestepIndex << " records.\n";
    // Save the truncated XML
    write();
}

bool PVDWriter::write() {
    std::string fileName = base_.string() + ".pvd";
    auto success = doc_.SaveFile(fileName.c_str());
    return success == tinyxml2::XML_SUCCESS;
}

} // namespace tndm
