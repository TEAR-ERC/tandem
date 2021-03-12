#ifndef BANNER_20210312_H
#define BANNER_20210312_H

#include "util/Affinity.h"

#include <cstddef>
#include <ostream>
#include <string_view>

namespace tndm {

class Banner {
public:
    static constexpr char Logo[] = R"LOGO(
               ___          ___         _____         ___          ___
      ___     /  /\        /__/\       /  /::\       /  /\        /__/\
     /  /\   /  /::\       \  \:\     /  /:/\:\     /  /:/_      |  |::\
    /  /:/  /  /:/\:\       \  \:\   /  /:/  \:\   /  /:/ /\     |  |:|:\
   /  /:/  /  /:/~/::\  _____\__\:\ /__/:/ \__\:| /  /:/ /:/_  __|__|:|\:\
  /  /::\ /__/:/ /:/\:\/__/::::::::\\  \:\ /  /://__/:/ /:/ /\/__/::::| \:\
 /__/:/\:\\  \:\/:/__\/\  \:\~~\~~\/ \  \:\  /:/ \  \:\/:/ /:/\  \:\~~\__\/
 \__\/  \:\\  \::/      \  \:\  ~~~   \  \:\/:/   \  \::/ /:/  \  \:\
      \  \:\\  \:\       \  \:\        \  \::/     \  \:\/:/    \  \:\
       \__\/ \  \:\       \  \:\        \__\/       \  \::/      \  \:\
              \__\/        \__\/                     \__\/        \__\/
)LOGO";

    static void print_logo(std::ostream& out) { out << Logo; }
    static void print_centered(std::ostream& out, std::string_view str,
                               std::size_t length = std::string_view::npos);
    static void print_logo_and_version(std::ostream& out);
    static void print_logo_version_and_affinity(std::ostream& out, Affinity const& affinity);

private:
    static constexpr std::size_t logo_width();
};

}; // namespace tndm

#endif // BANNER_20210312_H
