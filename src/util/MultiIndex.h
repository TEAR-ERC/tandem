#ifndef MULTIINDEX_H
#define MULTIINDEX_H

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace tndm {

template <std::size_t D, typename T>
T flatten(std::array<T, D> const& idx, std::array<T, D> const& shape) {
    T flat = 0;
    for (std::ptrdiff_t d = D - 1; d >= 0; --d) {
        flat *= shape[d];
        flat += idx[d];
    }
    return flat;
}

template <std::size_t D, typename T>
std::array<T, D> unflatten(T idx, std::array<T, D> const& shape) {
    std::array<T, D> a;
    for (std::size_t d = 0; d < D; ++d) {
        a[d] = idx % shape[d];
        idx /= shape[d];
    }
    return a;
}

template <typename T, typename... Entries>
constexpr std::array<T, sizeof...(Entries)> make_index(Entries&&... entries) {
    return {static_cast<T>(entries)...};
}

template <std::size_t D, typename T, typename U>
std::array<U, D> permute(std::array<T, D> const& permutation, std::array<U, D> const& index) {
    std::array<U, D> a;
    for (std::size_t d = 0; d < D; ++d) {
        assert(permutation[d] < D);
        a[d] = index[permutation[d]];
    }
    return a;
}

template <std::size_t D, typename T>
std::array<T, D> operator+(std::array<T, D> const& lhs, std::array<T, D> const& rhs) {
    std::array<T, D> result;
    for (std::size_t d = 0; d < D; ++d) {
        result[d] = lhs[d] + rhs[d];
    }
    return result;
}

template <std::size_t D, typename T>
std::array<T, D> operator-(std::array<T, D> const& lhs, std::array<T, D> const& rhs) {
    std::array<T, D> result;
    for (std::size_t d = 0; d < D; ++d) {
        result[d] = lhs[d] - rhs[d];
    }
    return result;
}

template <typename T, std::size_t D> class MultiIndexIterator {
public:
    using iterator_category = std::input_iterator_tag;
    using value_type = std::array<T, D>;
    using difference_type = std::array<std::ptrdiff_t, D>;
    using pointer = value_type*;
    using reference = value_type&;

    MultiIndexIterator(std::array<T, D> i, std::array<T, D> shape) : i_(i), shape_(shape) {}

    bool operator!=(MultiIndexIterator<T, D> const& other) {
        return i_ != other.i_ || shape_ != other.shape_;
    }

    auto& operator++() {
        ++i_[0];
        int d = 0;
        while (i_[d] >= shape_[d] && d < D - 1) {
            i_[d++] = 0;
            ++i_[d];
        }
        return *this;
    }
    auto operator++(int) {
        MultiIndexIterator<T, D> copy = *this;
        ++(*this);
        return copy;
    }

    reference operator*() { return i_; }

private:
    std::array<T, D> i_;
    std::array<T, D> shape_;
};

template <typename T, std::size_t D> class product {
public:
    product(std::array<T, D> shape) : shape_(shape) {}

    auto begin() { return MultiIndexIterator<T, D>({}, shape_); }
    auto end() {
        auto i = std::array<T, D>{};
        i.back() = shape_.back();
        return MultiIndexIterator<T, D>(i, shape_);
    }

private:
    std::array<T, D> shape_;
};

} // namespace tndm

#endif // MULTIINDEX_H
