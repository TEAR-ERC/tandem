#include "Banner.h"
#include "config.h"

#include <algorithm>

namespace tndm {

constexpr std::size_t Banner::logo_width() {
    std::size_t max_width = 0;
    std::size_t width = 0;
    for (std::size_t i = 0; i < sizeof(Logo) / sizeof(char); ++i) {
        if (Logo[i] == '\n') {
            max_width = std::max(width, max_width);
            width = 0;
        } else {
            ++width;
        }
    }
    return std::max(width, max_width);
}

void Banner::print_centered(std::ostream& out, std::string_view str, std::size_t length) {
    if (length == std::string_view::npos) {
        length = str.size();
    }
    if (length < logo_width()) {
        std::size_t fill = (logo_width() - length) / 2;
        out << std::string(fill, ' ') << str << std::endl;
    } else {
        out << str << std::endl;
    }
}

void Banner::print_logo_and_version(std::ostream& out) {
    out << Logo << std::endl;
    print_centered(out, "tandem version " + std::string(VersionString));
    out << std::endl << std::endl << std::endl;
}

void Banner::print_logo_version_and_affinity(std::ostream& out, Affinity const& affinity) {
    out << Logo << std::endl;
    print_centered(out, "tandem version " + std::string(VersionString));
    out << std::endl;

    print_centered(out, "Worker affinity");
    auto mask = affinity.to_string(affinity.worker_mask());
    auto mask_view = std::string_view(mask);
    auto line_width = logo_width();
    line_width -= line_width % 11;
    auto num_lines = 1 + (mask_view.size() - 1) / line_width;
    if (num_lines == 1) {
        line_width = std::string_view::npos;
    }
    for (int i = 0; i < num_lines; ++i) {
        print_centered(out, mask_view.substr(i * line_width, line_width), line_width);
    }

    out << std::endl << std::endl;
}

} // namespace tndm
