#ifndef PVDWRITER_20201021_H
#define PVDWRITER_20201021_H

#include <tinyxml2.h>

#include <string>

namespace tndm {

class PVDWriter {
public:
    PVDWriter();

    /**
     * @brief Add a pvtu file to the time series.
     */
    void addTimestep(double time, std::string const& fileName);

    /**
     * @brief Write PVD to disk on rank 0.
     *
     * @param baseName File name without extension
     *
     * @return True if write was successful.
     */
    bool write(std::string const& baseName);

private:
    tinyxml2::XMLDocument doc_;
    tinyxml2::XMLElement* collection_;
};

} // namespace tndm

#endif // PVDWRITER_20201021_H
