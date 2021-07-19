#ifndef SEASOPERATOR_20201001_H
#define SEASOPERATOR_20201001_H

#include "form/BoundaryMap.h"
#include "geometry/Curvilinear.h"
#include "tandem/RateAndStateBase.h"

#include "interface/BlockVector.h"
#include "parallel/LocalGhostCompositeView.h"
#include "parallel/Scatter.h"
#include "parallel/SparseBlockVector.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"
#include "util/Range.h"
#include "util/Scratch.h"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace tndm {

template <typename LocalOperator, typename SeasAdapter> class SeasOperator {
public:
    using time_functional_t = typename SeasAdapter::time_functional_t;

    SeasOperator(std::unique_ptr<LocalOperator> localOperator,
                 std::unique_ptr<SeasAdapter> seas_adapter)
        : lop_(std::move(localOperator)), adapter_(std::move(seas_adapter)),
          scratch_(lop_->scratch_mem_size() + adapter_->scratch_mem_size(), ALIGNMENT),
          scatter_(adapter_->faultMap().scatter_plan()),
          ghost_(scatter_.recv_prototype<double>(lop_->block_size(), ALIGNMENT)) {

        scratch_.reset();
        adapter_->begin_preparation(numElements());
        for (std::size_t faultNo = 0, num = numElements(); faultNo < num; ++faultNo) {
            adapter_->prepare(faultNo, scratch_);
        }
        adapter_->end_preparation();

        scratch_.reset();
        lop_->begin_preparation(numLocalElements());
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto fctNo = adapter_->faultMap().fctNo(faultNo);
            lop_->prepare(faultNo, adapter_->topo().info(fctNo), scratch_);
        }
        lop_->end_preparation();
    }

    std::size_t block_size() const { return lop_->block_size(); }
    std::size_t numElements() const { return adapter_->faultMap().size(); }
    std::size_t numLocalElements() const { return adapter_->faultMap().local_size(); }
    MPI_Comm comm() const { return adapter_->topo().comm(); }
    BoundaryMap const& faultMap() const { return adapter_->faultMap(); }
    SeasAdapter& adapter() { return *adapter_; }
    SeasAdapter const& adapter() const { return *adapter_; }
    std::size_t number_of_local_dofs() const { return numLocalElements() * block_size(); }

    void initial_condition(BlockVector& vector) {
        auto access_handle = vector.begin_access();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto B = access_handle.subtensor(slice{}, faultNo);
            lop_->pre_init(faultNo, B, scratch_);
        }
        scatter_.begin_scatter(vector, ghost_);
        scatter_.wait_scatter();

        auto access_handle_readonly = vector.begin_access_readonly();
        auto block_view = LocalGhostCompositeView(access_handle_readonly, ghost_);
        adapter_->solve(0.0, block_view);

        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction(block_view);
        scratch_.reset();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch_);

            auto B = access_handle.subtensor(slice{}, faultNo);
            lop_->init(faultNo, traction, B, scratch_);
        }
        adapter_->end_traction();
        vector.end_access_readonly(access_handle_readonly);
        vector.end_access(access_handle);
    }

    void rhs(double time, BlockVector const& state, BlockVector& result) {
        scatter_.begin_scatter(state, ghost_);
        scatter_.wait_scatter();

        auto in_handle = state.begin_access_readonly();
        auto block_view = LocalGhostCompositeView(in_handle, ghost_);
        adapter_->solve(time, block_view);

        auto out_handle = result.begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction(block_view);
        VMax_ = 0.0;
        scratch_.reset();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch_);

            auto state_block = in_handle.subtensor(slice{}, faultNo);
            auto result_block = out_handle.subtensor(slice{}, faultNo);
            double VMax = lop_->rhs(faultNo, time, traction, state_block, result_block, scratch_);

            VMax_ = std::max(VMax_, VMax);
        }
        adapter_->end_traction();
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
    }

    void full_solve(double time, BlockVector const& state) {
        scatter_.begin_scatter(state, ghost_);
        scatter_.wait_scatter();

        auto in_handle = state.begin_access_readonly();
        auto block_view = LocalGhostCompositeView(in_handle, ghost_);
        adapter_->full_solve(time, block_view);
        state.end_access_readonly(in_handle);
    }

    template <typename Iterator>
    auto raw_state(BlockVector const& vector, Iterator first, Iterator last) {
        auto num_elements = std::distance(first, last);
        auto soln = FiniteElementFunction<DomainDimension - 1u>(
            lop_->space().clone(), LocalOperator::NumQuantities, num_elements);
        auto& values = soln.values();

        auto in_handle = vector.begin_access_readonly();
        std::size_t out_no = 0;
        for (; first != last; ++first) {
            std::size_t faultNo = *first;
            assert(faultNo < numLocalElements());

            auto value_matrix = values.subtensor(slice{}, slice{}, out_no++);
            auto state_block = in_handle.subtensor(slice{}, faultNo);
            std::ptrdiff_t k = 0;
            for (std::ptrdiff_t j = 0; j < value_matrix.shape(1); ++j) {
                for (std::ptrdiff_t i = 0; i < value_matrix.shape(0); ++i) {
                    value_matrix(i, j) = state_block(k++);
                }
            }
        }
        vector.end_access_readonly(in_handle);
        return soln;
    }

    auto raw_state(BlockVector const& vector) {
        auto range = Range<std::size_t>(0, numLocalElements());
        return raw_state(vector, range.begin(), range.end());
    }

    template <typename Iterator>
    auto state(BlockVector const& vector, Iterator first, Iterator last) {
        auto num_elements = std::distance(first, last);
        auto soln = lop_->state_prototype(num_elements);
        auto& values = soln.values();

        auto in_handle = vector.begin_access_readonly();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        auto block_view = LocalGhostCompositeView(in_handle, ghost_);
        adapter_->begin_traction(block_view);
        scratch_.reset();
        std::size_t out_no = 0;
        for (; first != last; ++first) {
            std::size_t faultNo = *first;
            assert(faultNo < numLocalElements());
            adapter_->traction(faultNo, traction, scratch_);

            auto value_matrix = values.subtensor(slice{}, slice{}, out_no++);
            auto state_block = in_handle.subtensor(slice{}, faultNo);
            lop_->state(faultNo, traction, state_block, value_matrix, scratch_);
        }
        adapter_->end_traction();
        vector.end_access_readonly(in_handle);
        return soln;
    }

    auto state(BlockVector const& vector) {
        auto range = Range<std::size_t>(0, numLocalElements());
        return state(vector, range.begin(), range.end());
    }

    void set_boundary(time_functional_t fun) { adapter_->set_boundary(std::move(fun)); }

    double VMax_local() const { return VMax_; }
    LocalOperator& lop() { return *lop_; }

private:
    std::unique_ptr<LocalOperator> lop_;
    std::unique_ptr<SeasAdapter> adapter_;
    Scratch<double> scratch_;
    double VMax_ = 0.0;
    Scatter scatter_;
    SparseBlockVector<double> ghost_;
};

} // namespace tndm

#endif // SEASOPERATOR_20201001_H
