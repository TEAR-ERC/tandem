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

} // namespace tndm

#endif // UTILITY_H
