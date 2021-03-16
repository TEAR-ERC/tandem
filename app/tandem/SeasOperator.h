#ifndef SEASOPERATOR_20201001_H
#define SEASOPERATOR_20201001_H

#include "form/BoundaryMap.h"
#include "geometry/Curvilinear.h"

#include "tandem/RateAndStateBase.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"
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
          scratch_(lop_->scratch_mem_size() + adapter_->scratch_mem_size(), ALIGNMENT) {

        scratch_.reset();
        adapter_->begin_preparation(numLocalElements());
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
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
    std::size_t numLocalElements() const { return adapter_->faultMap().size(); }
    MPI_Comm comm() const { return adapter_->topo().comm(); }
    BoundaryMap const& faultMap() const { return adapter_->faultMap(); }
    SeasAdapter const& adapter() const { return *adapter_; }

    template <class BlockVector> void initial_condition(BlockVector& vector) {
        auto access_handle = vector.begin_access();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto B = vector.get_block(access_handle, faultNo);
            lop_->pre_init(faultNo, B, scratch_);
        }
        vector.end_access(access_handle);

        adapter_->solve(0.0, vector);

        access_handle = vector.begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&vector, &access_handle](std::size_t faultNo) {
            return vector.get_block(const_cast<typename BlockVector::const_handle>(access_handle),
                                    faultNo);
        });
        scratch_.reset();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch_);

            auto B = vector.get_block(access_handle, faultNo);
            lop_->init(faultNo, traction, B, scratch_);
        }
        adapter_->end_traction();
        vector.end_access(access_handle);
    }

    template <typename BlockVector> void rhs(double time, BlockVector& state, BlockVector& result) {
        adapter_->solve(time, state);

        auto in_handle = state.begin_access_readonly();
        auto out_handle = result.begin_access();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&state, &in_handle](std::size_t faultNo) {
            return state.get_block(in_handle, faultNo);
        });
        VMax_ = 0.0;
        scratch_.reset();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch_);

            auto state_block = state.get_block(in_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            double VMax = lop_->rhs(faultNo, time, traction, state_block, result_block, scratch_);

            VMax_ = std::max(VMax_, VMax);
        }
        adapter_->end_traction();
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
    }

    template <typename BlockVector> auto state(BlockVector& vector) {
        auto soln = lop_->state_prototype(numLocalElements());
        auto& values = soln.values();

        auto in_handle = vector.begin_access_readonly();
        auto traction = Managed<Matrix<double>>(adapter_->traction_info());
        adapter_->begin_traction([&vector, &in_handle](std::size_t faultNo) {
            return vector.get_block(in_handle, faultNo);
        });
        scratch_.reset();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            adapter_->traction(faultNo, traction, scratch_);

            auto value_matrix = values.subtensor(slice{}, slice{}, faultNo);
            auto state_block = vector.get_block(in_handle, faultNo);
            lop_->state(faultNo, traction, state_block, value_matrix, scratch_);
        }
        adapter_->end_traction();
        vector.end_access_readonly(in_handle);
        return soln;
    }

    void set_boundary(time_functional_t fun) { adapter_->set_boundary(std::move(fun)); }

    double VMax() const { return VMax_; }
    LocalOperator& lop() { return *lop_; }

private:
    std::unique_ptr<LocalOperator> lop_;
    std::unique_ptr<SeasAdapter> adapter_;
    Scratch<double> scratch_;
    double VMax_ = 0.0;
};

} // namespace tndm

#endif // SEASOPERATOR_20201001_H
