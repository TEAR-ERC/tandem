#ifndef HASH_20200627_H
#define HASH_20200627_H

#include <cstdint>
#include <string>

namespace tndm {

constexpr uint64_t fnv1a(char const* s, std::size_t len) {
    return len > 0 ? (fnv1a(s, len - 1) ^ s[len - 1]) * 0x00000100000001b3 : 0xcbf29ce484222325;
}
constexpr uint64_t fnv1a(std::string const& s) { return fnv1a(s.data(), s.size()); }

constexpr uint64_t operator""_fnv1a(char const* s, std::size_t len) { return tndm::fnv1a(s, len); }
} // namespace tndm

#endif // HASH_20200627_H
