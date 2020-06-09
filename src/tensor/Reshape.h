#ifndef RESHAPE_20200609_H
#define RESHAPE_20200609_H

#include "Tensor.h"
#include "util/Utility.h"

#include <type_traits>

namespace tndm {

template <typename TensorType, typename... Sizes> auto reshape(TensorType& tensor, Sizes... sizes) {
    auto newSize = (sizes * ...);
    assert(newSize == tensor.size());
    using new_real_t = std::remove_pointer_t<decltype(tensor.data())>;
    return Tensor<new_real_t, sizeof...(Sizes)>(
        tensor.data(), {static_cast<typename TensorType::index_t>(sizes)...});
}

} // namespace tndm

#endif // RESHAPE_20200609_H
