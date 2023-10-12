#include "ScalarWriter.h"

#include <iomanip>
#include <ios>
#include <filesystem>
namespace fs = std::filesystem;

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
        fs::path pckp(file_name_);
        bool exists = fs::exists(pckp);
        if (exists) {
            // open existing file
            out_->open(file_name_, true);
        } else { // below is needed to support checkpointing
            // open new file
            out_->open(file_name_, false);
            // write header
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
