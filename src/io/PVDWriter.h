#ifndef PVDWRITER_20201021_H
#define PVDWRITER_20201021_H

#include <tinyxml2.h>

#include <filesystem>
#include <string_view>

namespace tndm {

class PVDWriter {
public:
    PVDWriter(std::string_view baseName);

    /**
     * @brief Add a pvtu file to the time series.
     */
    void addTimestep(double time, std::string_view fileName);

    /**
     * @brief Write PVD to disk on rank 0.
     *
     * @param baseName File name without extension
     *
     * @return True if write was successful.
     */
    bool write();

private:
    std::filesystem::path base_;
    tinyxml2::XMLDocument doc_;
    tinyxml2::XMLElement* collection_;
};

} // namespace tndm

#endif // PVDWRITER_20201021_H
