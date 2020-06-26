#ifndef SEQUENCE_20200625_H
#define SEQUENCE_20200625_H

#include <utility>

namespace tndm {

template <class T, T... As, T... Bs>
constexpr std::integer_sequence<T, As..., Bs...> concat(std::integer_sequence<T, As...>,
                                                        std::integer_sequence<T, Bs...>) {
    return {};
}

template <template <typename> typename Predicate, auto Start>
constexpr auto make_filtered_sequence_with_start() {
    return std::index_sequence<>{};
}

template <template <typename> typename Predicate, auto Start, typename Head, typename... Tail>
constexpr auto make_filtered_sequence_with_start() {
    if constexpr (Predicate<Head>::value) {
        return concat(std::index_sequence<Start>{},
                      make_filtered_sequence_with_start<Predicate, Start + 1u, Tail...>());
    } else {
        return make_filtered_sequence_with_start<Predicate, Start + 1u, Tail...>();
    }
}

template <template <typename> typename Predicate, typename... Entry>
constexpr auto make_filtered_sequence() {
    return make_filtered_sequence_with_start<Predicate, 0u, Entry...>();
}

} // namespace tndm

#endif // SEQUENCE_20200625_H
