#ifndef TABLETRUNCATE_20251102_H
#define TABLETRUNCATE_20251102_H

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace tndm {

/**
 * @brief Keep only the header and the first num_records data records of a text table file.
 *
 * Used after a checkpoint restart to drop records that were written after the checkpoint.
 * A missing file is not an error (writers create their files lazily on first output).
 * The truncated content is written to a temporary file that atomically replaces the original.
 *
 * @param file_name Table file
 * @param num_records Number of data records to keep
 * @param num_header_lines Number of lines preceding the first record (title + column header)
 */
inline void truncate_table_file(std::string const& file_name, std::size_t num_records,
                                std::size_t num_header_lines = 2) {
    if (!std::filesystem::exists(file_name)) {
        std::cout << "Not truncating " << file_name << ": file does not exist yet.\n";
        return;
    }

    std::size_t const keep = num_header_lines + num_records;
    std::string const tmp_name = file_name + ".tmp";
    std::size_t lines_read = 0;
    {
        std::ifstream in(file_name);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open " + file_name +
                                     " for truncation (checkpoint restart)");
        }
        std::ofstream out(tmp_name, std::ios::trunc);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open " + tmp_name +
                                     " for writing truncated content");
        }
        std::string line;
        while (lines_read < keep && std::getline(in, line)) {
            out << line << '\n';
            ++lines_read;
        }
        out.flush();
        if (!out) {
            throw std::runtime_error("Failed to write " + tmp_name);
        }
    }

    if (lines_read < keep) {
        std::cerr << "Warning: " << keep << " lines expected in " << file_name
                  << " when truncating (checkpoint restart), but read only " << lines_read << "\n";
    }

    std::error_code ec;
    std::filesystem::rename(tmp_name, file_name, ec);
    if (ec) {
        throw std::runtime_error("Failed to replace " + file_name +
                                 " by truncated copy: " + ec.message());
    }
    std::cout << "Done truncating " << file_name << " to " << num_records << " records.\n";
}

} // namespace tndm

#endif // TABLETRUNCATE_20251102_H
