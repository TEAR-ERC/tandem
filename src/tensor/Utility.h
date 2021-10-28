#ifndef UTILITY_20201103_H
#define UTILITY_20201103_H

#include "tensor/TensorBase.h"
#include "util/LinearAllocator.h"

namespace tndm {

template <typename Tensor>
auto make_scratch_tensor(LinearAllocator<typename Tensor::real_t>& scratch,
                         TensorBase<Tensor> const& info) {
    auto tensor = Tensor(nullptr, info);
    auto* memory = scratch.allocate(tensor.size());
    return Tensor(memory, info);
}

template <typename Tensor, typename... Shape>
auto make_scratch_tensor(LinearAllocator<typename Tensor::real_t>& scratch, Shape... shape) {
    auto tensor = Tensor(nullptr, shape...);
    auto* memory = scratch.allocate(tensor.size());
    return Tensor(memory, shape...);
}

} // namespace tndm

#endif // UTILITY_20201103_H
