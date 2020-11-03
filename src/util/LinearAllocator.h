#ifndef LINEARALLOCATOR_20200911_H
#define LINEARALLOCATOR_20200911_H

#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>

namespace tndm {

template <typename T> class LinearAllocator {
public:
    using value_type = T;

    LinearAllocator(value_type* start, value_type* end, std::size_t alignment = alignof(T)) noexcept
        : start_(start), end_(end), alignment_(alignment) {
        reset();
    }
    template <class U> LinearAllocator(LinearAllocator<U> const& allocator) = delete;
    ~LinearAllocator() noexcept {}

    value_type* allocate(std::size_t n) {
        bool isPowerOfTwo = (alignment_ > 0 && ((alignment_ & (alignment_ - 1)) == 0));
        if (!isPowerOfTwo) {
            throw std::bad_alloc();
        }

        std::size_t size = n * sizeof(T);
        std::size_t space = (end_ - current_) * sizeof(T);
        void* ptr = std::align(alignment_, size, reinterpret_cast<void*&>(current_), space);
        if (ptr) {
            current_ += n;
        } else {
            throw std::bad_alloc();
        }
        return static_cast<value_type*>(ptr);
    }
    void deallocate(value_type*, std::size_t) noexcept {}
    void reset() { current_ = start_; }

private:
    value_type* start_;
    value_type* end_;
    value_type* current_;
    std::size_t alignment_;
};

template <class T, class U> bool operator==(LinearAllocator<T> const&, LinearAllocator<U> const&) {
    return true;
}
template <class T, class U>
bool operator!=(LinearAllocator<T> const& x, LinearAllocator<U> const& y) {
    return !(x == y);
}

} // namespace tndm

#endif // LINEARALLOCATOR_20200911_H
