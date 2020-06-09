#ifndef TENSORVIEW_20200609_H
#define TENSORVIEW_20200609_H

#include "TensorBase.h"

#include <cstddef>

namespace tndm {

template <typename TensorType> class TensorView : public TensorBase<TensorView<TensorType>> {
public:
    using Base = TensorBase<TensorView<TensorType>>;
    using typename Base::index_t;
    using typename Base::multi_index_t;
    using typename Base::real_t;

    TensorView(real_t* memory, multi_index_t const& shape) : Base(shape) { data_ = memory; }

    real_t* data() { return data_; }
    real_t const* data() const { return data_; }

protected:
    real_t* data_;
};

namespace detail {
template <typename TensorType> struct traits<TensorView<TensorType>> : traits<TensorType> {};
} // namespace detail

} // namespace tndm

#endif // TENSORVIEW_20200609_H
