#include "TablePrinter.h"
#include <initializer_list>

namespace tndm {

TablePrinter::TablePrinter(std::ostream& out, std::initializer_list<int> widths,
                           std::initializer_list<std::string> names)
    : out_(&out), widths_(std::move(widths)), col_(0), old_precision_(out.precision()) {
    auto name = names.begin();

    int col = 0;
    for (auto& w : widths_) {
        if (name != names.end()) {
            w = std::max(w, 1 + static_cast<int>(name->size()));
            *out_ << std::setw(w) << *name++;
        } else {
            w = std::max(4, w);
            *out_ << std::setw(w) << col;
        }
        ++col;
    }
    for (; name != names.end(); ++name) {
        int w = std::max(widths_.back(), 1 + static_cast<int>(name->size()));
        widths_.emplace_back(w);
        *out_ << std::setw(w) << *name;
    }
    *out_ << std::endl;
    separator();
}

TablePrinter::~TablePrinter() {
    separator();
    *out_ << std::setprecision(old_precision_) << std::defaultfloat;
}

void TablePrinter::separator() {
    int total_w = std::accumulate(widths_.begin(), widths_.end(), 0);
    auto old_fill = out_->fill();
    *out_ << std::setw(total_w) << std::setfill('-') << "" << std::setfill(old_fill)
          << std::scientific << std::endl;
}

} // namespace tndm
