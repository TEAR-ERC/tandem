#ifndef LINEARALLOCATOR_20200911_H
#define LINEARALLOCATOR_20200911_H

#include <cstddef>

namespace tndm {

class LinearAllocator {
public:
    LinearAllocator(void* start, void* end)
        : start_(reinterpret_cast<char*>(start)), end_(reinterpret_cast<char*>(end)) {
        reset();
    }
    LinearAllocator(LinearAllocator const& allocator) = delete;
    ~LinearAllocator() {}

    void* allocate(std::size_t size, std::size_t alignment = 0);
    template <typename T> T* allocate(std::size_t num, std::size_t alignment = 0) {
        if (alignment == 0) {
            alignment = alignof(T);
        }
        return reinterpret_cast<T*>(allocate(num * sizeof(T), alignment));
    }
    void free(void* ptr) {}
    void reset() { current_ = start_; }

private:
    char* start_;
    char* end_;
    char* current_;
};

} // namespace tndm

#endif // LINEARALLOCATOR_20200911_H
