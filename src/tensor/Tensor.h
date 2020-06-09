#ifndef TENSOR_20200609_H
#define TENSOR_20200609_H

#include "TensorBase.h"
#include "TensorView.h"

#include <cstddef>

namespace tndm {

template <typename real_t, std::size_t Dim> class Tensor : public TensorBase<Tensor<real_t, Dim>> {
public:
    using Base = TensorBase<Tensor>;
    using Myself = Tensor<real_t, Dim>;
    using typename Base::index_t;
    using typename Base::multi_index_t;

    Tensor(multi_index_t const& shape) : Base(shape) { data_ = new real_t[Base::size()]; }
    template <typename OtherTensor> Tensor(TensorInfo<OtherTensor> const& info) : Base(info) {
        data_ = new real_t[Base::size()];
    }
    ~Tensor() { delete[] data_; }

    real_t* data() { return data_; }
    real_t const* data() const { return data_; }

    auto view() { return TensorView<Myself>(data_, Base::shape()); }

protected:
    real_t* data_;
};

namespace detail {

template <typename RealT, std::size_t D> struct traits<Tensor<RealT, D>> {
    using real_t = RealT;
    static constexpr std::size_t Dim = D;
};

} // namespace detail

template <typename T>
Tensor(TensorInfo<T> const&) -> Tensor<typename TensorInfo<T>::real_t, TensorInfo<T>::Dim>;

} // namespace tndm

#endif // TENSOR_20200609_H
