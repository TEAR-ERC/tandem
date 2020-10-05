#ifndef SEASOPERATOR_20201001_H
#define SEASOPERATOR_20201001_H

#include "common/CmdLine.h"
#include "form/BC.h"
#include "form/BoundaryMap.h"
#include "form/DGOperatorTopo.h"
#include "mesh/LocalSimplexMesh.h"
#include "tensor/Managed.h"
#include "tensor/Reshape.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace tndm {

template <typename FaultOperator> class SeasOperator {
public:
    SeasOperator(std::shared_ptr<DGOperatorTopo> topo, std::unique_ptr<FaultOperator> fop)
        : topo_(std::move(topo)), lop_(std::move(fop)), faultMap_(*topo_, BC::Fault) {
        std::size_t numFacets = topo_->numLocalFacets();
        scratch_mem_ = std::make_unique<char[]>(lop_->scratch_mem_size());
        auto scratch = make_scratch();
        lop_->begin_preparation(numLocalElements());
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto fctNo = faultMap_.fctNo(faultNo);
            lop_->prepare(faultNo, topo_->info(fctNo), scratch);
        }
        lop_->end_preparation();
    }

    std::size_t block_size() const { return lop_->block_size(); }
    std::size_t numLocalElements() const { return faultMap_.size(); }
    MPI_Comm comm() const { return topo_->comm(); }
    BoundaryMap const& faultMap() const { return faultMap_; }

    template <class BlockVector> void initial_condition(BlockVector& vector) {
        auto bs = lop_->block_size();
        auto scratch = make_scratch();
        auto access_handle = vector.begin_access();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto B = vector.get_block(access_handle, faultNo);
            lop_->initial(faultNo, B, scratch);
        }
        vector.end_access(access_handle);
    }

    template <class LinearSolver, class TimeSolver>
    void quasi_dynamic_rhs(LinearSolver& ls, TimeSolver& ts) {
        // lop_->set_slip();
        ls.solve();

        // auto bs = lop_->block_size();
        auto fbs = lop_->block_size();
        // auto U = Managed<Vector<double>>(bs);
        auto B = Managed<Vector<double>>(fbs);

        auto scratch = make_scratch();
        // ls.begin_solution();
        ts.begin_state();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            lop_->rhs(faultNo, B, scratch);
            ts.set_state(faultNo, B);
        }
        ts.end_state();
        // ls.end_solution();
    }

    template <typename BlockVector> auto state(BlockVector& vector) {
        auto soln = lop_->state_prototype(numLocalElements());
        auto& values = soln.values();
        auto value_matrix = reshape(values, lop_->block_size(), numLocalElements());

        auto access_handle = vector.begin_access_readonly();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto block = value_matrix.subtensor(slice{}, faultNo);
            vector.copy(access_handle, faultNo, block);
        }
        vector.end_access_readonly(access_handle);
        return soln;
    }

private:
    auto make_scratch() const {
        return LinearAllocator(scratch_mem_.get(), scratch_mem_.get() + lop_->scratch_mem_size());
    }
    std::shared_ptr<DGOperatorTopo> topo_;
    std::unique_ptr<FaultOperator> lop_;
    std::unique_ptr<char[]> scratch_mem_;
    BoundaryMap faultMap_;
};

} // namespace tndm

#endif // SEASOPERATOR_20201001_H
