#ifndef ALGORITHM_20200710_H
#define ALGORITHM_20200710_H

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

} // namespace tndm

#endif // ALGORITHM_20200710_H
