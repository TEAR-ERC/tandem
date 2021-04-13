#ifndef NODALINTERPOLATION_20210317_H
#define NODALINTERPOLATION_20210317_H

#include "basis/Nodal.h"
#include "form/RefElement.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <cassert>
#include <vector>

namespace tndm {

template <std::size_t D> class NodalInterpolation {
public:
    NodalInterpolation(unsigned max_degree, NodesFactory<D> const& nodesFactory,
                       std::size_t numQuantities, std::size_t alignment)
        : numQuantities_(numQuantities) {
        spaces_.reserve(max_degree + 1);
        for (unsigned d = 0; d <= max_degree; ++d) {
            spaces_.push_back(NodalRefElement<D>(d, nodesFactory, alignment));
        }
    }

    std::size_t alignment() const { return spaces_.back().alignment(); }
    std::size_t scratch_mem_size() const {
        auto max_nbf = spaces_.back().numBasisFunctions();
        return LinearAllocator<double>::allocation_size(max_nbf * max_nbf, alignment());
    }

    unsigned max_degree() const { return spaces_.size() - 1; }
    std::size_t block_size(unsigned degree) const {
        assert(degree < spaces_.size());
        return spaces_[degree].numBasisFunctions() * numQuantities_;
    }

    void assemble(unsigned to_degree, unsigned from_degree, Matrix<double>& I,
                  LinearAllocator<double>& scratch) const;

private:
    std::size_t numQuantities_;
    std::vector<NodalRefElement<D>> spaces_;
};

} // namespace tndm

#endif // NODALINTERPOLATION_20210317_H
