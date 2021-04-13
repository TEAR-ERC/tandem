#ifndef ALGORITHM_20200710_H
#define ALGORITHM_20200710_H

#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace tndm {

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
 * @param v vector on which the permutaion is applied
 * @param indices index vector
 */
template <typename T, typename U>
void apply_permutation(std::vector<T>& v, std::vector<U> indices) {
    using std::swap;
    for (size_t i = 0; i < indices.size(); i++) {
        auto current = i;
        while (i != indices[current]) {
            auto next = indices[current];
            swap(v[current], v[next]);
            indices[current] = current;
            current = next;
        }
        indices[current] = current;
    }
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
