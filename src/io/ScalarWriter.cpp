#include "ScalarWriter.h"

#include <iomanip>
#include <ios>

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
        out_->open(file_name_, true);
    }

    *out_ << time;
    for (auto scalar : scalars) {
        *out_ << scalar;
    }
    *out_ << endrow;
    out_->close();
}

} // namespace tndm
