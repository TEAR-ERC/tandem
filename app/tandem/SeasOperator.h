#ifndef SEASOPERATOR_20201001_H
#define SEASOPERATOR_20201001_H

#include "common/CmdLine.h"
#include "form/BC.h"
#include "form/BoundaryMap.h"
#include "form/DGOperatorTopo.h"
#include "mesh/LocalSimplexMesh.h"
#include "tensor/Managed.h"
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
        : topo_(std::move(topo)), fop_(std::move(fop)), faultMap_(*topo_, BC::Fault) {
        std::size_t numFacets = topo_->numLocalFacets();
        scratch_mem_ = std::make_unique<char[]>(fop_->scratch_mem_size());
        auto scratch = make_scratch();
        fop_->begin_preparation(numFaultFaces());
        for (std::size_t faultNo = 0, num = numFaultFaces(); faultNo < num; ++faultNo) {
            auto fctNo = faultMap_.fctNo(faultNo);
            fop_->prepare(faultNo, topo_->info(fctNo), scratch);
        }
        fop_->end_preparation();
    }

    std::size_t numFaultFaces() const { return faultMap_.size(); }

    template <class BlockVector> void initial_condition(BlockVector& vector) {
        auto bs = fop_->block_size();
        auto scratch = make_scratch();
        auto access_handle = vector.begin_access();
        for (std::size_t faultNo = 0, num = numFaultFaces(); faultNo < num; ++faultNo) {
            auto B = vector.get_block(access_handle, faultNo);
            fop_->initial(faultNo, B, scratch);
        }
        vector.end_access(access_handle);
    }

    template <class LinearSolver, class TimeSolver>
    void quasi_dynamic_rhs(LinearSolver& ls, TimeSolver& ts) {
        // lop_->set_slip();
        ls.solve();

        // auto bs = lop_->block_size();
        auto fbs = fop_->block_size();
        // auto U = Managed<Vector<double>>(bs);
        auto B = Managed<Vector<double>>(fbs);

        auto scratch = make_scratch();
        // ls.begin_solution();
        ts.begin_state();
        for (std::size_t faultNo = 0, num = numFaultFaces(); faultNo < num; ++faultNo) {
            fop_->rhs(faultNo, B, scratch);
            ts.set_state(faultNo, B);
        }
        ts.end_state();
        // ls.end_solution();
    }

private:
    auto make_scratch() const {
        return LinearAllocator(scratch_mem_.get(), scratch_mem_.get() + fop_->scratch_mem_size());
    }
    std::shared_ptr<DGOperatorTopo> topo_;
    std::unique_ptr<FaultOperator> fop_;
    std::unique_ptr<char[]> scratch_mem_;
    BoundaryMap faultMap_;
};

} // namespace tndm

#endif // SEASOPERATOR_20201001_H
