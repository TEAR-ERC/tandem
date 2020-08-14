#ifndef TRAITS_20200814_H
#define TRAITS_20200814_H

#include <string>
#include <type_traits>

namespace tndm {

template <typename T> struct is_string : public std::false_type {};
template <> struct is_string<std::string> : public std::true_type {};

} // namespace tndm

#endif // TRAITS_20200814_H
