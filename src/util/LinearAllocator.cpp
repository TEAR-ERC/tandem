#include "LinearAllocator.h"

#include <memory>
#include <new>

namespace tndm {

void* LinearAllocator::allocate(size_t size, size_t alignment) {
    if (alignment == 0) {
        alignment = alignof(char);
    }
    bool isPowerOfTwo = (alignment > 0 && ((alignment & (alignment - 1)) == 0));
    if (!isPowerOfTwo) {
        throw std::bad_alloc();
    }

    std::size_t space = end_ - current_;
    void* ptr = std::align(alignment, size, reinterpret_cast<void*&>(current_), space);
    if (ptr) {
        current_ += size;
    } else {
        throw std::bad_alloc();
    }
    return ptr;
}

} // namespace tndm
