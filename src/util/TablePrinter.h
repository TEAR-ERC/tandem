#ifndef TABLEPRINTER_20210916_H
#define TABLEPRINTER_20210916_H

#include <cstddef>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <numeric>
#include <ostream>
#include <vector>

namespace tndm {

/**
 * @brief Convenience class to print an ASCII table
 */
class TablePrinter {
public:
    /**
     * @brief Sets up table header
     *
     * The number of columns is given by max(widths.size(), names.size()).
     * If the size of widths exceeds the size of names then exceeding column names are set
     * to the respective column number. If the size of names exceeds the size of widths
     * then the widths.back() is taken as column width.
     */
    TablePrinter(std::ostream& out, std::initializer_list<int> widths,
                 std::initializer_list<std::string> names);
    ~TablePrinter();
    TablePrinter(TablePrinter&& other) = default;
    TablePrinter(TablePrinter const& other) = default;
    TablePrinter& operator=(TablePrinter&& other) = default;
    TablePrinter& operator=(TablePrinter const& other) = default;

    template <typename T> TablePrinter& operator<<(T const& item) {
        auto width = col_ < widths_.size() ? widths_[col_] : 0;
        *out_ << std::setw(width) << std::setprecision(width - 7) << std::scientific << item;
        if (++col_ >= widths_.size()) {
            *out_ << std::endl;
            col_ = 0;
        }
        return *this;
    }

private:
    void separator();

    std::ostream* out_ = nullptr;
    std::vector<int> widths_;
    std::size_t col_;
    std::streamsize old_precision_;
};

} // namespace tndm

#endif // TABLEPRINTER_20210916_H
