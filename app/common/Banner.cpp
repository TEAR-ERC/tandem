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

void Banner::print_centered(std::ostream& out, std::string_view str) {
    if (str.size() < logo_width()) {
        std::size_t fill = (logo_width() - str.size()) / 2;
        out << std::string(fill, ' ') << str << std::endl;
    } else {
        out << str << std::endl;
    }
}

void Banner::print_logo_and_version(std::ostream& out) {
    out << std::endl << Logo << std::endl;
    print_centered(out, "tandem version " + std::string(VersionString));
    out << std::endl << std::endl << std::endl;
}

void Banner::print_logo_version_and_affinity(std::ostream& out, Affinity const& affinity) {
    out << std::endl << Logo << std::endl;
    print_centered(out, "tandem version " + std::string(VersionString));
    out << std::endl;
    print_centered(out, "Worker affinity");
    print_centered(out, affinity.to_string(affinity.worker_mask()));
    out << std::endl << std::endl;
}

} // namespace tndm
