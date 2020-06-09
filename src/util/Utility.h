#ifndef UTILITY_H
#define UTILITY_H

#include <tuple>
#include <utility>

namespace tndm {
namespace detail {
template <template <std::size_t> typename T, std::size_t... Is>
auto NTuple(std::index_sequence<Is...>) {
    return std::tuple<T<Is>...>{};
}
template <template <std::size_t> typename T, std::size_t N> auto NTuple() {
    return NTuple<T>(std::make_index_sequence<N>{});
}
} // namespace detail

template <template <std::size_t> typename T, std::size_t N>
using ntuple_t = decltype(detail::NTuple<T, N>());

template <typename T, typename U>
using copy_const = std::conditional_t<std::is_const_v<T>, std::add_const_t<U>, U>;

} // namespace tndm

#endif // UTILITY_H
