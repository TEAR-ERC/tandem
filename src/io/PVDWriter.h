#ifndef PVDWRITER_20201021_H
#define PVDWRITER_20201021_H

#include <string_view>
#include <tinyxml2.h>
#ifdef EXPERIMENTAL_FS
#include <experimental/filesystem>
#else
#include <filesystem>
#endif

#ifdef EXPERIMENTAL_FS
namespace fs = std::experimental::filesystem;
#else
namespace fs = std::filesystem;
#endif

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
    fs::path base_;
    tinyxml2::XMLDocument doc_;
    tinyxml2::XMLElement* collection_;
};

} // namespace tndm

#endif // PVDWRITER_20201021_H
