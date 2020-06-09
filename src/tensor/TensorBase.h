#ifndef TENSORBASE_20200609_H
#define TENSORBASE_20200609_H

#include "TensorInfo.h"

#include <array>

namespace tndm {

template <typename Derived> class TensorBase : public TensorInfo<Derived> {
public:
    using Base = TensorInfo<Derived>;
    using typename Base::index_t;
    using typename Base::multi_index_t;
    using typename Base::real_t;

    TensorBase(multi_index_t const& shape) : Base(shape) { computeStride(); }
    template <typename OtherTensor> TensorBase(TensorInfo<OtherTensor> const& info) : Base(info) {
        computeStride();
    }

    index_t size() const { return Base::shape().back() * stride_.back(); }

    template <typename... Entry> real_t& operator()(Entry... entry) {
        static_assert(sizeof...(Entry) == Base::Dim);
        return static_cast<Derived*>(this)->data()[address(entry...)];
    }
    template <typename... Entry> real_t const& operator()(Entry... entry) const {
        static_assert(sizeof...(Entry) == Base::Dim);
        return static_cast<Derived const*>(this)->data()[address(entry...)];
    }

    real_t& operator()(multi_index_t const& entry) {
        return static_cast<Derived*>(this)->data()[address(entry)];
    }
    real_t const& operator()(multi_index_t const& entry) const {
        return static_cast<Derived const*>(this)->data()[address(entry)];
    }

protected:
    void computeStride() {
        stride_[0] = 1;
        for (index_t d = 1; d < Base::Dim; ++d) {
            stride_[d] = stride_[d - 1] * Base::shape()[d - 1];
        }
    }

    template <typename Head> index_t address(Head head) const {
        assert(head >= 0 && head < Base::shape()[Base::Dim - 1u]);
        return head * stride_[Base::Dim - 1u];
    }

    template <typename Head, typename... Tail> index_t address(Head head, Tail... tail) const {
        constexpr index_t d = (Base::Dim - 1u) - sizeof...(Tail);
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

    multi_index_t stride_;
};

} // namespace tndm

#endif // TENSORBASE_20200609_H
