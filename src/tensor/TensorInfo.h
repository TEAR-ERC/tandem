#ifndef TENSORINFO_20200609_H
#define TENSORINFO_20200609_H

#include <array>
#include <cstddef>

namespace tndm {

namespace detail {
template <typename T> struct traits;
}

template <typename Derived> class TensorInfo {
public:
    static constexpr std::size_t Dim = detail::traits<Derived>::Dim;
    using index_t = std::ptrdiff_t;
    using multi_index_t = std::array<index_t, Dim>;
    using real_t = typename detail::traits<Derived>::real_t;

    TensorInfo(multi_index_t const& shape) : shape_(shape) {}
    template <typename OtherTensor>
    TensorInfo(TensorInfo<OtherTensor> const& info) : shape_(info.shape()) {}

    multi_index_t const& shape() const { return shape_; }
    index_t shape(index_t pos) const { return shape_[pos]; }

protected:
    multi_index_t shape_;
};

} // namespace tndm

#endif // TENSORINFO_20200609_H
