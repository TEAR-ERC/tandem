#ifndef SCALARWRITER_20210721_H
#define SCALARWRITER_20210721_H

#include "TableWriter.h"

#include <mneme/span.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tndm {

class ScalarWriter {
public:
    ScalarWriter(std::string_view prefix, std::unique_ptr<TableWriter> table_writer,
                 std::vector<std::string> variable_names)
        : file_name_(prefix), out_(std::move(table_writer)),
          variable_names_(std::move(variable_names)) {
        file_name_ += out_->default_extension();
    }

    void write(double time, mneme::span<double> scalars) const;

private:
    void write_header() const;

    std::string file_name_;
    std::unique_ptr<TableWriter> out_;
    std::vector<std::string> variable_names_;
};

} // namespace tndm

#endif // SCALARWRITER_20210721_H
