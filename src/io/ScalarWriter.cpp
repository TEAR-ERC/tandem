#include "ScalarWriter.h"

#include <iomanip>
#include <ios>
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

void ScalarWriter::write_header() const {
    out_->add_title("Scalar output");
    *out_ << beginheader << "Time";
    for (auto variable_name : variable_names_) {
        *out_ << variable_name;
    }
    *out_ << endheader;
}

void ScalarWriter::write(double time, mneme::span<double> scalars) const {
    if (time <= 0.0) {
        out_->open(file_name_, false);
        write_header();
    } else {
        if (fs::exists(file_name_)) {
            out_->open(file_name_, true);
        } else {
            out_->open(file_name_, false);
            write_header();
        }
    }

    *out_ << time;
    for (auto scalar : scalars) {
        *out_ << scalar;
    }
    *out_ << endrow;
    out_->close();
}

} // namespace tndm
