#ifndef MANAGED_20200609_H
#define MANAGED_20200609_H

#include "Tensor.h"
#include "TensorBase.h"

#include <cstddef>
#include <memory>

namespace tndm {

template <typename Tensor> class Managed : public Tensor {
public:
    using typename Tensor::index_t;
    using typename Tensor::multi_index_t;
    using typename Tensor::real_t;

    Managed(multi_index_t const& shape)
        : Tensor(nullptr, shape), managedData(std::make_unique<real_t[]>(Tensor::size())) {
        Tensor::data_ = managedData.get();
    }

    template <typename... Shape>
    Managed(Shape... shape)
        : Tensor(nullptr, shape...), managedData(std::make_unique<real_t[]>(Tensor::size())) {
        Tensor::data_ = managedData.get();
    }

    // Declare but not define copy constructor to satisfy compiler (we use copy elision)
    Managed(Managed<Tensor> const& other);
    Managed& operator=(Managed<Tensor> const& other);

    Managed(Managed<Tensor>&& other) = default;
    Managed& operator=(Managed<Tensor>&& other) = default;

    template <typename OtherTensor>
    Managed(TensorBase<OtherTensor> const& info)
        : Tensor(nullptr, info), managedData(std::make_unique<real_t[]>(Tensor::size())) {
        Tensor::data_ = managedData.get();
    }
    virtual ~Managed() = default;

protected:
    std::unique_ptr<real_t[]> managedData;
};

template <typename T>
Managed(TensorBase<T> const&)
    -> Managed<Tensor<typename detail::traits<T>::real_t, detail::traits<T>::Dim>>;

namespace detail {
template <typename Tensor> struct traits<Managed<Tensor>> : public traits<Tensor> {};
} // namespace detail

} // namespace tndm

#endif // MANAGED_20200609_H
