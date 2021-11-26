#include "Banner.h"
#include "config.h"

#include <algorithm>
#include <sstream>
#include <sys/resource.h>

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

void Banner::print_logo(std::ostream& out) { out << Logo; }

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

void Banner::print_version(std::ostream& out) {
    print_centered(out, "tandem version " + std::string(VersionString));
}

void Banner::print_stack_limit(std::ostream& out) {
    constexpr auto warning_limit = 64; // MiB

    std::stringstream s;
    s << "stack size limit = ";
    bool warning = false;
    rlimit rlim;
    if (getrlimit(RLIMIT_STACK, &rlim) == 0) {
        if (rlim.rlim_cur == RLIM_INFINITY) {
            s << "unlimited";
        } else {
            const auto rlim_MiB = rlim.rlim_cur / 1024 / 1024;
            s << rlim_MiB << " MiB";
            warning = (rlim_MiB < warning_limit);
        }
    } else {
        s << "unknown";
    }
    print_centered(out, s.str());
    if (warning) {
        s.str(std::string());
        s << "it with 'ulimit -Ss " << warning_limit * 1024 << "' or 'ulimit -Ss unlimited'.";
        print_centered(out, "WARNING: Your stack size limit is quite small, increase");
        print_centered(out, s.str());
    }
}

void Banner::print_affinity(std::ostream& out, Affinity const& affinity) {
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
}

void Banner::standard(std::ostream& out, Affinity const& affinity) {
    print_logo(out);
    out << std::endl;
    print_version(out);
    out << std::endl;
    print_stack_limit(out);
    out << std::endl;
    print_affinity(out, affinity);
    out << std::endl << std::endl;
}

} // namespace tndm
