#ifndef ENUMERATE_20200610_H
#define ENUMERATE_20200610_H

#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>

namespace tndm {

template <typename Iterator> class EnumerateIterator {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using iterator_reference = decltype(*std::declval<Iterator>());
    using value_type = std::pair<std::size_t, iterator_reference>;
    using reference = value_type&;

    EnumerateIterator(Iterator&& iterator, std::size_t beginCount = 0)
        : iterator(iterator), count(beginCount) {}

    bool operator!=(EnumerateIterator const& other) { return iterator != other.iterator; }
    bool operator==(EnumerateIterator const& other) { return !(*this != other); }

    EnumerateIterator& operator++() {
        ++count;
        ++iterator;
        return *this;
    }
    EnumerateIterator& operator--() {
        --count;
        --iterator;
        return *this;
    }
    EnumerateIterator operator++(int) {
        Iterator copy(*this);
        ++(*this);
        return copy;
    }
    EnumerateIterator operator--(int) {
        Iterator copy(*this);
        --(*this);
        return copy;
    }

    value_type operator*() { return std::make_pair(count, *iterator); }

private:
    Iterator iterator;
    std::size_t count = std::numeric_limits<std::size_t>::max();
};

template <typename Iterable> class enumerate {
public:
    enumerate(Iterable&& iterable, std::size_t beginCount = 0u)
        : iterable(std::forward<Iterable>(iterable)), beginCount(beginCount) {}

    auto begin() { return EnumerateIterator(iterable.begin(), beginCount); }
    auto end() { return EnumerateIterator(iterable.end()); }

private:
    Iterable iterable;
    std::size_t beginCount;
};

template <typename Iterable> enumerate(Iterable&&) -> enumerate<Iterable>;

} // namespace tndm

#endif // ENUMERATE_20200610_H
