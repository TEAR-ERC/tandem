#ifndef ADAPTEROPERATOR_20210906_H
#define ADAPTEROPERATOR_20210906_H

#include "config.h"
#include "form/AbstractAdapterOperator.h"

#include "form/BoundaryMap.h"
#include "interface/BlockVector.h"
#include "interface/BlockView.h"
#include "localoperator/Adapter.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"
#include "util/Scratch.h"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace tndm {

template <typename LocalOperator> class AdapterOperator : public AbstractAdapterOperator {
public:
    AdapterOperator(std::shared_ptr<LocalOperator> adapted_lop,
                    std::unique_ptr<Adapter<LocalOperator>> lop,
                    std::shared_ptr<DGOperatorTopo> topo, std::shared_ptr<BoundaryMap> fault_map)
        : adapted_lop_(std::move(adapted_lop)), lop_(std::move(lop)), topo_(std::move(topo)),
          fault_map_(std::move(fault_map)), scratch_(lop_->scratch_mem_size(), ALIGNMENT) {

        scratch_.reset();
        lop_->begin_preparation(num_elements());
        for (std::size_t faultNo = 0, num = num_elements(); faultNo < num; ++faultNo) {
            auto fctNo = fault_map_->fctNo(faultNo);
            lop_->prepare(faultNo, topo_->info(fctNo), scratch_);
        }
        lop_->end_preparation();
    }

    BoundaryMap const& fault_map() const override { return *fault_map_; }
    std::size_t num_local_elements() const override { return fault_map_->local_size(); }
    std::size_t traction_block_size() const override { return lop_->traction_block_size(); }
    MPI_Comm comm() const override { return topo_->comm(); }
    std::size_t num_elements() const { return fault_map_->size(); }
    Adapter<LocalOperator> const& lop() const { return *lop_; }

    auto slip_bc(BlockView const& state)
        -> std::function<void(std::size_t, Matrix<double>&, bool)> override {
        return [this, &state](std::size_t fctNo, Matrix<double>& f_q, bool) {
            auto faultNo = fault_map_->bndNo(fctNo);
            auto state_block = state.get_block(faultNo);
            lop_->slip(faultNo, state_block, f_q);
        };
    }

    void traction(BlockView const& displacement, BlockVector& result) override {
        auto traction_q = Managed<Matrix<double>>(adapted_lop_->tractionResultInfo().shape(),
                                                  std::size_t{ALIGNMENT});

        scratch_.reset();
        auto result_handle = result.begin_access();
        for (std::size_t faultNo = 0, num = num_local_elements(); faultNo < num; ++faultNo) {
            auto fctNo = fault_map_->fctNo(faultNo);
            auto const& info = topo_->info(fctNo);
            auto u0 = displacement.get_block(info.up[0]);
            auto u1 = displacement.get_block(info.up[1]);
            if (info.up[0] == info.up[1]) {
                adapted_lop_->traction_boundary(fctNo, info, u0, traction_q);
            } else {
                adapted_lop_->traction_skeleton(fctNo, info, u0, u1, traction_q);
            }

            auto result_block = result_handle.subtensor(slice{}, faultNo);
            lop_->traction(faultNo, traction_q, result_block, scratch_);
        }
        result.end_access(result_handle);
    }

    void slip_rate(std::size_t faultNo, Vector<double const>& slip_rate,
                   Matrix<double>& slip_rate_q) override {
        lop_->slip_rate(faultNo, slip_rate, slip_rate_q);
    }
    void moment_rate(std::size_t faultNo, Matrix<double>& moment_rate_vector,
                     Matrix<double>& slip_rate_q, std::size_t fctNo,
                     FacetInfo const& info) override {
        std::size_t nq = slip_rate_q.shape(1);
        alignas(ALIGNMENT) double mu_field_raw[nq];
        auto mu_field = Matrix<double>(mu_field_raw, 1, nq);
        adapted_lop_->mu_avg(fctNo, info, mu_field);
        lop_->moment_rate(faultNo, moment_rate_vector, slip_rate_q, mu_field);
    }

private:
    std::shared_ptr<LocalOperator> adapted_lop_;
    std::unique_ptr<Adapter<LocalOperator>> lop_;
    std::shared_ptr<DGOperatorTopo> topo_;
    std::shared_ptr<BoundaryMap> fault_map_;
    Scratch<double> scratch_;
};

} // namespace tndm

#endif // ADAPTEROPERATOR_20210906_H
