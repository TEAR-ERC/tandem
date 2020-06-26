#ifndef SEQUENCE_20200625_H
#define SEQUENCE_20200625_H

#include <utility>

template <class T, T... As, T... Bs>
constexpr std::integer_sequence<T, As..., Bs...> operator+(std::integer_sequence<T, As...>,
                                                           std::integer_sequence<T, Bs...>) {
    return {};
}

namespace tndm {

template <template <typename> typename Predicate, auto Start>
constexpr auto make_filtered_sequence_with_start() {
    return std::index_sequence<>{};
}

template <template <typename> typename Predicate, auto Start, typename Head, typename... Tail>
constexpr auto make_filtered_sequence_with_start() {
    if constexpr (Predicate<Head>::value) {
        return std::index_sequence<Start>{} +
               make_filtered_sequence_with_start<Predicate, Start + 1, Tail...>();
    } else {
        return make_filtered_sequence_with_start<Predicate, Start + 1, Tail...>();
    }
}

template <template <typename> typename Predicate, typename... Entry>
constexpr auto make_filtered_sequence() {
    return make_filtered_sequence_with_start<Predicate, 0, Entry...>();
}

} // namespace tndm

#endif // SEQUENCE_20200625_H
