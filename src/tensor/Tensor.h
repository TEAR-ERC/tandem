#ifndef TENSOR_20200609_H
#define TENSOR_20200609_H

#include "TensorBase.h"

#include <array>

namespace tndm {

template <typename RealT, std::size_t D> class Tensor : public TensorBase<Tensor<RealT, D>> {
public:
    using Base = TensorBase<Tensor<RealT, D>>;
    using typename Base::index_t;
    using typename Base::multi_index_t;
    using real_t = RealT;

    Tensor(real_t* memory, multi_index_t const& shape) : Base(shape), data_(memory) {
        computeStride();
    }

    template <typename... Shape>
    Tensor(real_t* memory, Shape... shape) : Base(shape...), data_(memory) {
        computeStride();
    }

    template <typename OtherTensor>
    Tensor(real_t* memory, TensorBase<OtherTensor> const& info) : Base(info), data_(memory) {
        computeStride();
    }
    virtual ~Tensor() = default;

    index_t size() const { return Base::shape().back() * stride_.back(); }

    template <typename... Entry> real_t& operator()(Entry... entry) {
        static_assert(sizeof...(Entry) == D);
        return data_[address(entry...)];
    }
    template <typename... Entry> real_t const& operator()(Entry... entry) const {
        static_assert(sizeof...(Entry) == D);
        return data_[address(entry...)];
    }

    real_t& operator()(multi_index_t const& entry) { return data_[address(entry)]; }
    real_t const& operator()(multi_index_t const& entry) const { return data_[address(entry)]; }

    real_t* data() { return data_; }
    real_t const* data() const { return data_; }

protected:
    void computeStride() {
        stride_[0] = 1;
        for (index_t d = 1; d < D; ++d) {
            stride_[d] = stride_[d - 1] * Base::shape()[d - 1];
        }
    }

    template <typename Head> index_t address(Head head) const {
        assert(head >= 0 && head < Base::shape()[D - 1u]);
        return head * stride_[D - 1u];
    }

    template <typename Head, typename... Tail> index_t address(Head head, Tail... tail) const {
        constexpr index_t d = (D - 1u) - sizeof...(Tail);
        assert(head >= 0 && head < Base::shape()[d]);
        return head * stride_[d] + address(tail...);
    }

    index_t address(multi_index_t const& entry) const {
        index_t addr = 0;
        for (index_t d = 0; d < entry.size(); ++d) {
            addr += entry[d] * stride_[d];
        }
        return addr;
    }

    real_t* data_;
    multi_index_t stride_;
};

template <typename real_t> using Vector = Tensor<real_t, 1u>;
template <typename real_t> using Matrix = Tensor<real_t, 2u>;

namespace detail {

template <typename RealT, std::size_t D> struct traits<Tensor<RealT, D>> {
    using real_t = RealT;
    static constexpr std::size_t Dim = D;
};

template <typename RealT, std::size_t D> struct traits<const Tensor<RealT, D>> {
    using real_t = const RealT;
    static constexpr std::size_t Dim = D;
};

} // namespace detail

} // namespace tndm

#endif // TENSOR_20200609_H
