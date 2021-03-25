#ifndef SPARSEBLOCKVECTOR_20210325_H
#define SPARSEBLOCKVECTOR_20210325_H

#include "tensor/Tensor.h"

#include <algorithm>
#include <limits>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <vector>

namespace tndm {

template <typename T> class SparseBlockVector {
public:
    static constexpr std::size_t DefaultAlignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;

    SparseBlockVector(std::vector<std::size_t> local_to_global, std::size_t block_size = 1,
                      std::size_t alignment = DefaultAlignment)
        : block_size_(block_size) {
        mem_ = make_storage(local_to_global.size() * block_size_, alignment);

        auto max_idx = std::max_element(local_to_global.begin(), local_to_global.end());
        if (max_idx != local_to_global.end()) {
            global_to_local_.resize(*max_idx + 1, std::numeric_limits<std::size_t>::max());
        }
        std::size_t local = 0;
        for (auto&& i : local_to_global) {
            global_to_local_[i] = local++;
        }
    }

    T* data() { return mem_.get(); }
    std::size_t block_size() const { return block_size_; }

    bool has_block(std::size_t idx) const {
        return idx < global_to_local_.size() &&
               global_to_local_[idx] != std::numeric_limits<std::size_t>::max();
    }

    auto get_block(std::size_t idx) {
        assert(has_block(idx));
        return Vector<T>(&mem_[global_to_local_[idx] * block_size_], block_size_);
    }

    auto get_block(std::size_t idx) const {
        assert(has_block(idx));
        return Vector<T const>(&mem_[global_to_local_[idx] * block_size_], block_size_);
    }

private:
    struct Deleter {
        void operator()(T* ptr) { std::free(ptr); }
    };
    std::unique_ptr<T[], Deleter> make_storage(std::size_t num_T, std::size_t alignment) const {
        return std::unique_ptr<T[], Deleter>(
            static_cast<T*>(std::aligned_alloc(alignment, sizeof(T) * num_T)), Deleter{});
    }

    std::unique_ptr<T[], Deleter> mem_;
    std::size_t block_size_;
    std::vector<std::size_t> global_to_local_;
};

} // namespace tndm

#endif // SPARSEBLOCKVECTOR_20210325_H
