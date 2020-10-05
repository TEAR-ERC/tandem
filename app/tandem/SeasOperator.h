#ifndef SEASOPERATOR_20201001_H
#define SEASOPERATOR_20201001_H

#include "common/CmdLine.h"
#include "form/BC.h"
#include "form/DGOperatorTopo.h"
#include "mesh/LocalSimplexMesh.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <petscvec.h>
#include <utility>
#include <vector>

namespace tndm {

template <typename SpatialOperator, typename FaultOperator> class SeasOperator {
public:
    SeasOperator(std::shared_ptr<DGOperatorTopo> topo, std::unique_ptr<SpatialOperator> lop,
                 std::unique_ptr<FaultOperator> fop)
        : topo_(std::move(topo)), fop_(std::move(fop)) {
        std::size_t numFacets = topo_->numLocalFacets();
        fctNos_.reserve(numFacets);
        faultNos_.resize(numFacets, std::numeric_limits<std::size_t>::max());
        for (std::size_t fctNo = 0; fctNo < numFacets; ++fctNo) {
            auto const& info = topo_->info(fctNo);
            if (info.bc == BC::Fault) {
                faultNos_[fctNo] = fctNos_.size();
                fctNos_.push_back(fctNo);
            }
        }
        fctNos_.shrink_to_fit();

        fault_scratch_mem_ = std::make_unique<char[]>(fop_->scratch_mem_size());
        auto scratch = make_fault_scratch();
        fop_->begin_preparation(numFaultFaces());
        for (std::size_t faultNo = 0, num = numFaultFaces(); faultNo < num; ++faultNo) {
            auto fctNo = fctNos_[faultNo];
            fop_->prepare(faultNo, topo_->info(fctNo), scratch);
        }
        fop_->end_preparation();
    }

    std::size_t numFaultFaces() const { return fctNos_.size(); }

    template <class LinearSolver, class TimeSolver>
    void setup_quasi_dynamic(LinearSolver& ls, TimeSolver& ts) {
        // base::assemble(ls, false);

        auto fbs = fop_->block_size();
        ts.create_state(fbs, numFaultFaces(), false, topo_->comm());

        auto scratch = make_fault_scratch();
        ts.begin_state();
        for (std::size_t faultNo = 0, num = numFaultFaces(); faultNo < num; ++faultNo) {
            auto B = ts.get_state(faultNo);
            fop_->initial(faultNo, B, scratch);
        }
        ts.end_state();
    }

    template <class LinearSolver, class TimeSolver>
    void quasi_dynamic_rhs(LinearSolver& ls, TimeSolver& ts) {
        // lop_->set_slip();
        ls.solve();

        // auto bs = lop_->block_size();
        auto fbs = fop_->block_size();
        // auto U = Managed<Vector<double>>(bs);
        auto B = Managed<Vector<double>>(fbs);

        auto scratch = make_fault_scratch();
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
    auto make_fault_scratch() const {
        return LinearAllocator(fault_scratch_mem_.get(),
                               fault_scratch_mem_.get() + fop_->scratch_mem_size());
    }
    std::shared_ptr<DGOperatorTopo> topo_;
    std::unique_ptr<FaultOperator> fop_;
    std::unique_ptr<char[]> fault_scratch_mem_;
    std::vector<std::size_t> fctNos_;
    std::vector<std::size_t> faultNos_;
};

} // namespace tndm

#endif // SEASOPERATOR_20201001_H
