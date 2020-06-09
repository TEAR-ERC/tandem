#ifndef TENSORBASE_20200609_H
#define TENSORBASE_20200609_H

#include <array>
#include <cstddef>

namespace tndm {

namespace detail {
template <typename T> struct traits;
}

template <typename Derived> class TensorBase {
public:
    using index_t = std::ptrdiff_t;
    using multi_index_t = std::array<index_t, detail::traits<Derived>::Dim>;

    TensorBase(multi_index_t const& shape) : shape_(shape) {}

    template <typename... Shape>
    TensorBase(Shape... shape) : shape_{static_cast<index_t>(shape)...} {}

    template <typename OtherTensor>
    TensorBase(TensorBase<OtherTensor> const& info) : shape_(info.shape()) {}

    virtual ~TensorBase() = default;

    multi_index_t const& shape() const { return shape_; }
    index_t shape(index_t pos) const { return shape_[pos]; }

protected:
    multi_index_t shape_;
};

} // namespace tndm

#endif // TENSORBASE_20200609_H
