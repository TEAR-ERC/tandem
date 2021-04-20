#ifndef PROBE_20210419_H
#define PROBE_20210419_H

#include <array>
#include <cstddef>
#include <string>

namespace tndm {

template <std::size_t D> struct Probe {
    std::string name;
    std::array<double, D> x;
};

} // namespace tndm

#endif // PROBE_20210419_H
