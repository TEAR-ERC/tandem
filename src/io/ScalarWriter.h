#ifndef SCALARWRITER_20210721_H
#define SCALARWRITER_20210721_H

#include <mneme/span.hpp>

#include <cstddef>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace tndm {

class ScalarWriter {
public:
    ScalarWriter(std::string_view prefix, std::vector<std::string> variable_names)
        : file_name_(prefix), variable_names_(std::move(variable_names)) {
        file_name_ += ".dat";
    }

    void write(double time, mneme::span<double> scalars) const;

private:
    void write_header(std::ofstream& file) const;

    std::string file_name_;
    std::vector<std::string> variable_names_;
};

} // namespace tndm

#endif // SCALARWRITER_20210721_H
