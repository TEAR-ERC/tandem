#ifndef ADAPTEROPERATOR_20210906_H
#define ADAPTEROPERATOR_20210906_H

#include "config.h"

#include "form/BoundaryMap.h"
#include "interface/BlockVector.h"
#include "interface/BlockView.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"
#include "util/Scratch.h"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace tndm {

template <typename LocalOperator> class AdapterOperator {
public:
    AdapterOperator(std::unique_ptr<LocalOperator> lop, std::shared_ptr<DGOperatorTopo> topo,
                    std::shared_ptr<BoundaryMap> fault_map)
        : lop_(std::move(lop)), topo_(std::move(topo)), fault_map_(std::move(fault_map)),
          scratch_(lop_->scratch_mem_size(), ALIGNMENT) {

        scratch_.reset();
        lop_->begin_preparation(num_elements());
        for (std::size_t faultNo = 0, num = num_elements(); faultNo < num; ++faultNo) {
            auto fctNo = fault_map_->fctNo(faultNo);
            lop_->prepare(faultNo, topo_->info(fctNo), scratch_);
        }
        lop_->end_preparation();
    }

    std::size_t num_elements() const { return fault_map_->size(); }
    std::size_t num_local_elements() const { return fault_map_->local_size(); }
    BoundaryMap const& fault_map() const { return *fault_map_; }
    LocalOperator const& lop() const { return *lop_; }
    MPI_Comm comm() const { return topo_->comm(); }

    auto slip_bc(BlockView const& state) {
        return [this, &state](std::size_t fctNo, Matrix<double>& f_q, bool) {
            auto faultNo = fault_map_->bndNo(fctNo);
            auto state_block = state.get_block(faultNo);
            lop_->slip(faultNo, state_block, f_q);
        };
    }

    template <typename DGOp>
    void traction(DGOp& dgop, BlockView const& displacement, BlockVector& result) {
        auto traction_q = Managed<Matrix<double>>(dgop.lop().tractionResultInfo().shape(),
                                                  std::size_t{ALIGNMENT});

        scratch_.reset();
        auto result_handle = result.begin_access();
        for (std::size_t faultNo = 0, num = num_local_elements(); faultNo < num; ++faultNo) {
            auto fctNo = fault_map_->fctNo(faultNo);
            auto const& info = topo_->info(fctNo);
            auto u0 = displacement.get_block(info.up[0]);
            auto u1 = displacement.get_block(info.up[1]);
            if (info.up[0] == info.up[1]) {
                dgop.lop().traction_boundary(fctNo, info, u0, traction_q);
            } else {
                dgop.lop().traction_skeleton(fctNo, info, u0, u1, traction_q);
            }

            auto result_block = result_handle.subtensor(slice{}, faultNo);
            lop_->traction(faultNo, traction_q, result_block, scratch_);
        }
        result.end_access(result_handle);
    }

private:
    std::unique_ptr<LocalOperator> lop_;
    std::shared_ptr<DGOperatorTopo> topo_;
    std::shared_ptr<BoundaryMap> fault_map_;
    Scratch<double> scratch_;
};

} // namespace tndm

#endif // ADAPTEROPERATOR_20210906_H
