#ifndef MODALINTERPOLATION_20210318_H
#define MODALINTERPOLATION_20210318_H

#include "form/RefElement.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <cassert>
#include <vector>

namespace tndm {

template <std::size_t D> class ModalInterpolation {
public:
    ModalInterpolation(unsigned max_degree, std::size_t numQuantities, std::size_t alignment)
        : numQuantities_(numQuantities) {
        spaces_.reserve(max_degree + 1);
        for (unsigned d = 0; d <= max_degree; ++d) {
            spaces_.push_back(ModalRefElement<D>(d, alignment));
        }
    }

    std::size_t alignment() const { return spaces_.back().alignment(); }
    std::size_t scratch_mem_size() const { return 0; }

    unsigned max_degree() const { return spaces_.size() - 1; }
    std::size_t block_size(unsigned degree) const {
        assert(degree < spaces_.size());
        return spaces_[degree].numBasisFunctions() * numQuantities_;
    }

    void assemble(unsigned to_degree, unsigned from_degree, Matrix<double>& I,
                  LinearAllocator<double>&) const {
        std::size_t N1 = spaces_[to_degree].numBasisFunctions();
        std::size_t N2 = spaces_[from_degree].numBasisFunctions();

        assert(I.shape(0) == N1 * numQuantities);
        assert(I.shape(1) == N2 * numQuantities);

        I.set_zero();
        for (int l = 0; l < N2; ++l) {
            for (int p = 0; p < numQuantities_; ++p) {
                I(l + p * N1, l + p * N2) = 1.0;
            }
        }
    }

private:
    std::vector<ModalRefElement<D>> spaces_;
    std::size_t numQuantities_;
};

} // namespace tndm

#endif // MODALINTERPOLATION_20210318_H
