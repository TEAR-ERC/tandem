#ifndef INTERPOLATIONOPERATOR_20210317_H
#define INTERPOLATIONOPERATOR_20210317_H

#include "tensor/Reshape.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"
#include "util/Scratch.h"

#include <experimental/type_traits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tndm {

template <typename LocalOperator> class InterpolationOperator {
public:
    using local_operator_t = LocalOperator;

    InterpolationOperator(std::size_t numLocalElements, std::unique_ptr<LocalOperator> lop)
        : numLocalElements_(numLocalElements), lop_(std::move(lop)) {}

    unsigned max_degree() const { return lop_->max_degree(); }
    std::size_t block_size(unsigned degree) const { return lop_->block_size(degree); }

    template <typename BlockMatrix>
    void assemble(unsigned to_degree, unsigned from_degree, BlockMatrix& matrix) {
        if (to_degree <= from_degree) {
            throw std::logic_error("to_degree must be larger than from_degree.");
        }

        auto bs_lp1 = lop_->block_size(to_degree);
        auto bs_l = lop_->block_size(from_degree);
        std::size_t I_size =
            LinearAllocator<double>::allocation_size(bs_lp1 * bs_l, lop_->alignment());
        auto scratch = Scratch<double>(I_size + lop_->scratch_mem_size(), lop_->alignment());

        matrix.begin_assembly();
        for (std::size_t elNo = 0; elNo < numLocalElements_; ++elNo) {
            scratch.reset();
            double* buffer = scratch.allocate(bs_lp1 * bs_l);
            auto I = Matrix<double>(buffer, bs_lp1, bs_l);
            lop_->assemble(to_degree, from_degree, I, scratch);
            matrix.add_block(elNo, elNo, I);
        }
        matrix.end_assembly();
    }

private:
    std::size_t numLocalElements_;
    std::unique_ptr<LocalOperator> lop_;
};

} // namespace tndm

#endif // INTERPOLATIONOPERATOR_20210317_H
