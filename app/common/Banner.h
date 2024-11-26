#ifndef BANNER_20210312_H
#define BANNER_20210312_H

#include "parallel/Affinity.h"

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

    static void print_logo(std::ostream& out);
    static void print_centered(std::ostream& out, std::string_view str,
                               std::size_t length = std::string_view::npos);
    static void print_version(std::ostream& out);
    static void print_stack_limit(std::ostream& out);
    static void print_affinity(std::ostream& out, Affinity const& affinity,
                               std::string_view node_mask);

    static void standard(std::ostream& out, Affinity const& affinity, std::string_view node_mask);

private:
    static constexpr std::size_t logo_width();
};

}; // namespace tndm

#endif // BANNER_20210312_H
