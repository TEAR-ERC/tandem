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

    /**
     * @brief Allocation size measured in sizeof(T).
     *
     * Remarks:
     * 1. It is assumed that "start" is properly aligned.
     * 2. The allocation size includes the padding for the next allocation. E.g.
     *    if a 24 byte allocation is made with 32 byte alignment, then the next
     *    allocation on the same buffer must start at 32 byte offset and not at
     *    24 byte offset.
     *
     * @param n Number of T
     * @param alignment Alignment in bytes
     */
    static std::size_t allocation_size(std::size_t n, std::size_t alignment) {
        std::size_t size = n * sizeof(T);
        std::size_t a_size = (1 + (size - 1) / alignment) * alignment;
        return 1 + (a_size - 1) / sizeof(T);
    }

    LinearAllocator(value_type* start, value_type* end, std::size_t alignment = alignof(T))
        : start_(start), end_(end), alignment_(alignment) {
        if (reinterpret_cast<uintptr_t>(start) % alignment != 0) {
            throw std::invalid_argument("Start pointer given to LinerAllocator must satisfy the "
                                        "requested memory alignment.");
        }
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

protected:
    void set_pointers(value_type* start, value_type* end) {
        start_ = start;
        end_ = end;
        reset();
    }

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
