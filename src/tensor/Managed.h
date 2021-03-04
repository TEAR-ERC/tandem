#ifndef MANAGED_20200609_H
#define MANAGED_20200609_H

#include "Tensor.h"
#include "TensorBase.h"

#include <cstddef>
#include <cstdlib>
#include <memory>

namespace tndm {

template <typename Tensor> class Managed : public Tensor {
public:
    using typename Tensor::index_t;
    using typename Tensor::multi_index_t;
    using typename Tensor::real_t;
    static constexpr std::size_t DefaultAlignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;

    Managed() : Tensor(nullptr, multi_index_t{}) {}
    Managed(multi_index_t const& shape, std::size_t alignment = DefaultAlignment)
        : Tensor(nullptr, shape), managedData(make_storage(Tensor::size(), alignment)) {
        Tensor::data_ = managedData.get();
    }
    template <typename... Shape>
    Managed(Shape... shape)
        : Tensor(nullptr, shape...), managedData(make_storage(Tensor::size(), DefaultAlignment)) {
        Tensor::data_ = managedData.get();
    }

    // Declare but not define copy constructor to satisfy compiler (we use copy elision)
    Managed(Managed<Tensor> const& other);
    Managed& operator=(Managed<Tensor> const& other);

    Managed(Managed<Tensor>&& other) = default;
    Managed& operator=(Managed<Tensor>&& other) = default;

    template <typename OtherTensor>
    Managed(TensorBase<OtherTensor> const& info, std::size_t alignment = DefaultAlignment)
        : Tensor(nullptr, info), managedData(make_storage(Tensor::size(), alignment)) {
        Tensor::data_ = managedData.get();
    }
    virtual ~Managed() = default;

protected:
    struct Deleter {
        void operator()(real_t* ptr) { std::free(ptr); }
    };

    std::unique_ptr<real_t[], Deleter> make_storage(std::size_t num_reals,
                                                    std::size_t alignment) const {
        return std::unique_ptr<real_t[], Deleter>(
            static_cast<real_t*>(std::aligned_alloc(alignment, sizeof(real_t) * num_reals)),
            Deleter{});
    }

    std::unique_ptr<real_t[], Deleter> managedData;
};

template <typename T>
Managed(TensorBase<T> const&)
    -> Managed<Tensor<typename detail::traits<T>::real_t, detail::traits<T>::Dim>>;

namespace detail {
template <typename Tensor> struct traits<Managed<Tensor>> : public traits<Tensor> {};
template <typename Tensor> struct traits<const Managed<Tensor>> : public traits<const Tensor> {};
} // namespace detail

} // namespace tndm

#endif // MANAGED_20200609_H
