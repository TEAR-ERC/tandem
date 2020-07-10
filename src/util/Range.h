#ifndef RANGE_20200710_H
#define RANGE_20200710_H

#include <cstddef>
#include <iterator>

namespace tndm {

template <typename Int> class RangeIterator;

template <typename IntT> struct Range {
    Range(IntT to) : from(0), to(to) {}
    Range(IntT from, IntT to) : from(from), to(to) {}

    IntT size() const noexcept { return to - from; }
    IntT length() const noexcept { return size(); }
    RangeIterator<IntT> begin() { return RangeIterator(from); }
    RangeIterator<IntT> end() { return RangeIterator(to); }

    IntT from;
    IntT to;
};

template <typename IntT> class RangeIterator {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = IntT;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    RangeIterator(IntT pos) : pos(pos) {}

    bool operator!=(RangeIterator const& other) { return pos != other.pos; }
    bool operator==(RangeIterator const& other) { return !(*this != other); }

    auto& operator++() {
        ++pos;
        return *this;
    }
    auto& operator--() {
        --pos;
        return *this;
    }
    auto operator++(int) {
        RangeIterator copy(pos);
        ++pos;
        return copy;
    }
    auto operator--(int) {
        RangeIterator copy(pos);
        --pos;
        return copy;
    }

    value_type operator*() { return pos; }

private:
    IntT pos;
};

} // namespace tndm

#endif // RANGE_20200710_H
