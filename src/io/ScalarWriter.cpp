#include "ScalarWriter.h"

#include <iomanip>
#include <ios>

namespace tndm {

void ScalarWriter::write_header(std::ofstream& file) const {
    file << "TITLE = \"Scalar output\"" << std::endl;
    file << "VARIABLES = \"Time\"";
    for (auto variable_name : variable_names_) {
        file << ",\"" << variable_name << "\"";
    }
    file << std::endl;
}

void ScalarWriter::write(double time, mneme::span<double> scalars) const {
    std::ofstream file;
    if (time <= 0.0) {
        file.open(file_name_, std::ios::out);
        write_header(file);
    } else {
        file.open(file_name_, std::ios::app);
    }

    file << std::scientific << std::setprecision(15);
    file << time;
    for (auto scalar : scalars) {
        file << " " << scalar;
    }
    file << std::endl;
    file.close();
}

} // namespace tndm
