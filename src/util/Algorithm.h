#ifndef ALGORITHM_20200710_H
#define ALGORITHM_20200710_H

#include "tensor/Tensor.h"
#include "util/MultiIndex.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace tndm {
namespace detail {

template <typename Array, typename U> class SwapArray {
public:
    SwapArray(Array& a) : a_(a) {}
    void operator()(U i, U j) const { std::swap(a_[i], a_[j]); }

private:
    Array& a_;
};

template <typename Tensor, typename U> class SwapTensor {
public:
    SwapTensor(Tensor& tensor, std::size_t mode)
        : data_(tensor.data()), mode_stride_(tensor.stride(mode)),
          reduced_shape_(remove_mode(tensor.shape(), mode)),
          reduced_stride_(remove_mode(tensor.stride(), mode)) {}

    void operator()(U i, U j) const {
        for (auto const& entry : product(reduced_shape_)) {
            std::size_t idx = 0;
            for (std::size_t z = 0; z < entry.size(); ++z) {
                idx += entry[z] * reduced_stride_[z];
            }
            std::swap(data_[idx + i * mode_stride_], data_[idx + j * mode_stride_]);
        }
    }

private:
    using real_t = typename Tensor::real_t;
    using multi_t = typename Tensor::multi_index_t;
    using reduced_t = std::array<typename Tensor::index_t, detail::traits<Tensor>::Dim - 1>;

    static auto remove_mode(multi_t const& a, std::size_t mode) {
        reduced_t r;
        std::copy(a.begin(), a.begin() + mode, r.begin());
        std::copy(a.begin() + mode + 1, a.end(), r.begin() + mode);
        return r;
    }

    real_t* data_;
    std::size_t mode_stride_;
    reduced_t reduced_shape_;
    reduced_t reduced_stride_;
};

template <typename Swap, typename U> void apply_permutation(Swap&& swap, std::vector<U>&& indices) {
    for (size_t i = 0; i < indices.size(); i++) {
        auto current = i;
        while (i != indices[current]) {
            auto next = indices[current];
            swap(current, next);
            indices[current] = current;
            current = next;
        }
        indices[current] = current;
    }
}

} // namespace detail

/**
 * @brief Apply permutation to a vector
 *
 * Indices is passed by copy as the algorithm reorders the indices array.
 * If you do not need indices anymore use
 * apply_permutation(v, std::move(indices));
 * to save the copy.
 *
 * Source: https://devblogs.microsoft.com/oldnewthing/20170102-00/?p=95095
 *
 * @tparam T type stored in vector
 * @tparam U index type
 * @param v vector on which the permutation is applied
 * @param indices index vector
 */
template <typename T, typename U>
void apply_permutation(std::vector<T>& v, std::vector<U> indices) {
    assert(v.size() == indices.size());
    detail::apply_permutation(detail::SwapArray<std::vector<T>, U>(v), std::move(indices));
}

/**
 * @brief Apply permutation to a tensor on a mode
 *
 * Example: Let the order-3 tensor A and the permutation p be given.
 * For the resulting tensor B we have:
 * mode=0: B_{i,:,:} = A_{p[i],:,:}
 * mode=1: B_{:,i,:} = A_{:,p[i],:}
 * mode=2: B_{:,:,i} = A_{:,:,p[i]}
 *
 * @tparam Tensor tensor type
 * @tparam U index type
 * @param t tensor on which the permutation is applied
 * @param indices index vector
 * @param mode tensor mode that is permuted
 */
template <typename Tensor, typename U>
void apply_permutation(Tensor& t, std::vector<U> indices, std::size_t mode) {
    assert(t.shape(mode) == indices.size());
    detail::apply_permutation(detail::SwapTensor<Tensor, U>(t, mode), std::move(indices));
}

/**
 * @brief Finds consecutive blocks in a list of indices
 *
 * @tparam IterableType Some iterable type, e.g. std::vector or std::array
 * @tparam OutIdxT Integer type of output
 * @param indices Vector of indices
 *
 * @return block_lengths and displacements
 */
template <typename IterableType, typename OutIdxT = typename IterableType::value_type>
auto find_blocks(IterableType const& indices)
    -> std::tuple<std::vector<OutIdxT>, std::vector<OutIdxT>> {
    std::vector<OutIdxT> block_lengths, displacements;

    block_lengths.reserve(indices.size());
    displacements.reserve(indices.size());
    for (auto idx : indices) {
        if (!displacements.empty() && !block_lengths.empty() &&
            idx - displacements.back() == block_lengths.back()) {
            ++block_lengths.back();
        } else {
            block_lengths.push_back(1);
            displacements.push_back(idx);
        }
    }
    block_lengths.shrink_to_fit();
    displacements.shrink_to_fit();

    return {block_lengths, displacements};
}

} // namespace tndm

#endif // ALGORITHM_20200710_H
