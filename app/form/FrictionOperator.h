#ifndef FRICTIONOPERATOR_20210906_H
#define FRICTIONOPERATOR_20210906_H

#include "config.h"
#include "form/AbstractFrictionOperator.h"

#include "form/BoundaryMap.h"
#include "form/DGOperatorTopo.h"
#include "form/FiniteElementFunction.h"
#include "interface/BlockVector.h"
#include "tensor/Reshape.h"
#include "tensor/Tensor.h"
#include "util/Range.h"
#include "util/Scratch.h"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace tndm {

template <typename LocalOperator> class FrictionOperator : public AbstractFrictionOperator {
public:
    FrictionOperator(std::unique_ptr<LocalOperator> lop, std::shared_ptr<DGOperatorTopo> topo,
                     std::shared_ptr<BoundaryMap> fault_map)
        : lop_(std::move(lop)), topo_(std::move(topo)), fault_map_(std::move(fault_map)),
          scratch_(lop_->scratch_mem_size(), ALIGNMENT) {
        scratch_.reset();
        lop_->begin_preparation(num_local_elements());
        for (std::size_t faultNo = 0, num = num_local_elements(); faultNo < num; ++faultNo) {
            auto fctNo = fault_map_->fctNo(faultNo);
            lop_->prepare(faultNo, topo_->info(fctNo), scratch_);
        }
        lop_->end_preparation();
    }

    std::size_t block_size() const override { return lop_->block_size(); }
    std::size_t slip_block_size() const override { return lop_->slip_block_size(); }
    std::size_t num_local_elements() const override { return fault_map_->local_size(); }
    double VMax_local() const override { return VMax_; }
    std::size_t num_elements() const { return fault_map_->size(); }
    MPI_Comm comm() const { return topo_->comm(); }
    BoundaryMap const& fault_map() const { return *fault_map_; }
    LocalOperator& lop() { return *lop_; }

    void pre_init(BlockVector& state) override {
        auto state_handle = state.begin_access();
        for (std::size_t faultNo = 0, num = num_local_elements(); faultNo < num; ++faultNo) {
            auto state_block = state_handle.subtensor(slice{}, faultNo);
            lop_->pre_init(faultNo, state_block, scratch_);
        }
        state.end_access(state_handle);
    }

    void init(double time, BlockVector const& traction, BlockVector& state) override {
        auto traction_handle = traction.begin_access_readonly();
        auto state_handle = state.begin_access();
        VMax_ = 0.0;
        scratch_.reset();
        for (std::size_t faultNo = 0, num = num_local_elements(); faultNo < num; ++faultNo) {
            auto traction_block = traction_handle.subtensor(slice{}, faultNo);
            auto state_block = state_handle.subtensor(slice{}, faultNo);
            double VMax = lop_->init(time, faultNo, traction_block, state_block, scratch_);

            VMax_ = std::max(VMax_, VMax);
        }
        state.end_access(state_handle);
        traction.end_access_readonly(traction_handle);
    }

    void rhs(double time, BlockVector const& traction, BlockVector const& state,
             BlockVector& result) override {
        auto traction_handle = traction.begin_access_readonly();
        auto state_handle = state.begin_access_readonly();
        auto result_handle = result.begin_access();
        VMax_ = 0.0;
        scratch_.reset();
        for (std::size_t faultNo = 0, num = num_local_elements(); faultNo < num; ++faultNo) {
            auto traction_block = traction_handle.subtensor(slice{}, faultNo);
            auto state_block = state_handle.subtensor(slice{}, faultNo);
            auto result_block = result_handle.subtensor(slice{}, faultNo);
            double VMax =
                lop_->rhs(time, faultNo, traction_block, state_block, result_block, scratch_);

            VMax_ = std::max(VMax_, VMax);
        }
        result.end_access(result_handle);
        state.end_access_readonly(state_handle);
        traction.end_access_readonly(traction_handle);
    }

    template <typename Iterator>
    auto raw_state(BlockVector const& state, Iterator first, Iterator last) {
        auto num_elements = std::distance(first, last);
        auto soln = FiniteElementFunction<DomainDimension - 1u>(
            lop_->space().clone(), LocalOperator::NumQuantities, num_elements);
        auto& values = soln.values();

        auto in_handle = state.begin_access_readonly();
        std::size_t out_no = 0;
        for (; first != last; ++first) {
            std::size_t faultNo = *first;
            assert(faultNo < num_local_elements());

            auto value_matrix = values.subtensor(slice{}, slice{}, out_no++);
            auto state_block = in_handle.subtensor(slice{}, faultNo);
            auto state_matrix = reshape(state_block, value_matrix.shape(0), value_matrix.shape(1));
            value_matrix.copy_values(state_matrix);
        }
        state.end_access_readonly(in_handle);
        return soln;
    }

    auto raw_state(BlockVector const& vector)
        -> FiniteElementFunction<DomainDimension - 1u> override {
        auto range = Range<std::size_t>(0, num_local_elements());
        return raw_state(vector, range.begin(), range.end());
    }

    template <typename Iterator>
    auto state(double time, BlockVector const& traction, BlockVector const& state, Iterator first,
               Iterator last) {
        auto num_elements = std::distance(first, last);
        auto soln = lop_->state_prototype(num_elements);
        auto& values = soln.values();

        auto traction_handle = traction.begin_access_readonly();
        auto state_handle = state.begin_access_readonly();
        scratch_.reset();
        std::size_t out_no = 0;
        for (; first != last; ++first) {
            std::size_t faultNo = *first;
            assert(faultNo < num_local_elements());

            auto traction_block = traction_handle.subtensor(slice{}, faultNo);
            auto state_block = state_handle.subtensor(slice{}, faultNo);
            auto value_matrix = values.subtensor(slice{}, slice{}, out_no++);
            lop_->state(time, faultNo, traction_block, state_block, value_matrix, scratch_);
        }
        state.end_access_readonly(state_handle);
        traction.end_access_readonly(traction_handle);
        return soln;
    }

    auto state(double time, BlockVector const& traction, BlockVector const& state_vec,
               std::vector<std::size_t> const& subset)
        -> FiniteElementFunction<DomainDimension - 1u> override {
        return state(time, traction, state_vec, subset.begin(), subset.end());
    }

    auto state(double time, BlockVector const& traction, BlockVector const& state_vec,
               std::optional<Range<std::size_t>> range = std::nullopt)
        -> FiniteElementFunction<DomainDimension - 1u> override {
        if (!range) {
            *range = Range<std::size_t>(0, num_local_elements());
        }
        return state(time, traction, state_vec, range->begin(), range->end());
    }

    template <typename Iterator> auto params(Iterator first, Iterator last) {
        auto num_elements = std::distance(first, last);
        auto soln = lop_->params_prototype(num_elements);
        auto& values = soln.values();

        scratch_.reset();
        std::size_t out_no = 0;
        for (; first != last; ++first) {
            std::size_t faultNo = *first;
            assert(faultNo < num_local_elements());

            auto value_matrix = values.subtensor(slice{}, slice{}, out_no++);
            lop_->params(faultNo, value_matrix, scratch_);
        }
        return soln;
    }

    auto params() {
        auto range = Range<std::size_t>(0, num_local_elements());
        return params(range.begin(), range.end());
    }

private:
    std::unique_ptr<LocalOperator> lop_;
    std::shared_ptr<DGOperatorTopo> topo_;
    std::shared_ptr<BoundaryMap> fault_map_;
    Scratch<double> scratch_;
    double VMax_ = 0.0;
};

} // namespace tndm

#endif // FRICTIONOPERATOR_20210906_H
