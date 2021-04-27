#ifndef TENSOR_20200609_H
#define TENSOR_20200609_H

#include "TensorBase.h"
#include "util/Sequence.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <type_traits>

namespace tndm {

struct slice {};

template <typename RealT, std::size_t D, bool Packed = true>
class Tensor : public TensorBase<Tensor<RealT, D>> {
public:
    using Base = TensorBase<Tensor<RealT, D>>;
    using typename Base::index_t;
    using typename Base::multi_index_t;
    using real_t = RealT;

    template <typename T> using is_slice = std::is_same<slice, T>;

    Tensor() : Base(), data_(nullptr) {}

    Tensor(real_t* memory, multi_index_t const& shape) : Base(shape), data_(memory) {
        static_assert(Packed);
        computeStride();
    }

    Tensor(real_t* memory, multi_index_t const& shape, multi_index_t const& stride)
        : Base(shape), data_(memory), stride_(stride) {
        static_assert(!Packed);
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

    template <typename... Entry> auto subtensor(Entry... entry) {
        return extractSubtensor<real_t, Entry...>(entry...);
    }

    template <typename... Entry> auto subtensor(Entry... entry) const {
        return extractSubtensor<real_t, Entry...>(entry...);
    }

    multi_index_t const& stride() const { return stride_; }
    index_t stride(index_t pos) const { return stride_[pos]; }

    void set_zero() { std::fill(data_, data_ + size(), RealT(0.0)); }

    template <typename OtherRealT, bool OtherPacked>
    void copy_values(Tensor<OtherRealT, D, OtherPacked> const& other) {
        multi_index_t entry{};
        auto stop0 = this->shape(0);
        while (entry[D - 1] != this->shape(D - 1)) {
            OtherRealT const* source = &other(entry);
            real_t* target = &operator()(entry);
            for (index_t i = 0; i < stop0; ++i) {
                *target = *source;
                target += stride_[0];
                source += other.stride(0);
            }

            if (D == 1) {
                break;
            }

            index_t d = 0;
            do {
                entry[d] = 0;
                d++;
                ++entry[d];
            } while (entry[d] == this->shape(d) && d < D - 1);
        }
    }

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

    template <std::size_t... Is>
    std::array<index_t, sizeof...(Is)> extractNumbers(multi_index_t const& multiIndex,
                                                      std::index_sequence<Is...>) const {
        return {multiIndex[Is]...};
    }

    template <typename new_real_t, typename... Entry> auto extractSubtensor(Entry... entry) const {
        constexpr auto filtered_sequence = make_filtered_sequence<is_slice, Entry...>();
        auto filter_entry = [](auto entry) {
            if constexpr (is_slice<decltype(entry)>::value) {
                return 0;
            } else {
                return entry;
            }
        };
        auto addr = address(filter_entry(entry)...);
        constexpr bool subPacked =
            Packed && std::is_same_v<std::decay_t<decltype(filtered_sequence)>,
                                     std::make_index_sequence<filtered_sequence.size()>>;
        using subtensor_t =
            Tensor<std::remove_pointer_t<new_real_t>, filtered_sequence.size(), subPacked>;
        if constexpr (subPacked) {
            return subtensor_t(data_ + addr, extractNumbers(Base::shape(), filtered_sequence));
        } else {
            return subtensor_t(data_ + addr, extractNumbers(Base::shape(), filtered_sequence),
                               extractNumbers(stride_, filtered_sequence));
        }
    }

    real_t* data_;
    multi_index_t stride_;
};

template <typename real_t, typename T>
Tensor(real_t*, TensorBase<T> const&) -> Tensor<real_t, detail::traits<T>::Dim>;

template <typename real_t> using Vector = Tensor<real_t, 1u>;
template <typename real_t> using Matrix = Tensor<real_t, 2u>;

namespace detail {

template <typename RealT, std::size_t D, bool Packd> struct traits<Tensor<RealT, D, Packd>> {
    using real_t = RealT;
    static constexpr std::size_t Dim = D;
    static constexpr bool Packed = Packd;
};

template <typename RealT, std::size_t D, bool Packd> struct traits<const Tensor<RealT, D, Packd>> {
    using real_t = const RealT;
    static constexpr std::size_t Dim = D;
    static constexpr bool Packed = Packd;
};

} // namespace detail

} // namespace tndm

#endif // TENSOR_20200609_H
