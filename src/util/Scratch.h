#ifndef SCRATCH_20210316_H
#define SCRATCH_20210316_H

#include "util/LinearAllocator.h"

#include <cstddef>
#include <cstdlib>
#include <memory>

namespace tndm {

template <typename T> class Scratch : public LinearAllocator<T> {
public:
    Scratch(std::size_t num_T, std::size_t alignment = alignof(T))
        : LinearAllocator<T>(nullptr, nullptr, alignment), mem_(make_storage(num_T, alignment)) {
        this->set_pointers(mem_.get(), mem_.get() + num_T);
    }

private:
    struct Deleter {
        void operator()(T* ptr) { std::free(ptr); }
    };

    std::unique_ptr<T[], Deleter> make_storage(std::size_t num_T, std::size_t alignment) const {
        std::size_t size = sizeof(T) * num_T;
        size = (1 + (size - 1) / alignment) * alignment;
        return std::unique_ptr<T[], Deleter>(static_cast<T*>(std::aligned_alloc(alignment, size)),
                                             Deleter{});
    }

    std::unique_ptr<T[], Deleter> mem_;
};

} // namespace tndm

#endif // SCRATCH_20210316_H
