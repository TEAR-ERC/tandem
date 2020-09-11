#ifndef DGOPERATOR_20200909_H
#define DGOPERATOR_20200909_H

#include "form/DGOperatorTopo.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <memory>
#include <utility>

namespace tndm {

template <std::size_t D, typename LocalOperator> class DGOperator : public DGOperatorTopo<D> {
public:
    using base = DGOperatorTopo<D>;

    DGOperator(LocalSimplexMesh<D> const& mesh, std::unique_ptr<LocalOperator> lop, MPI_Comm comm)
        : DGOperatorTopo<D>(mesh, comm), lop_(std::move(lop)) {
        scratch_mem_ = std::make_unique<char[]>(lop_->scratch_mem_size());

        auto scratch =
            LinearAllocator(scratch_mem_.get(), scratch_mem_.get() + lop_->scratch_mem_size());
        lop_->begin_preparation(base::numElements(), base::numLocalElements(),
                                base::numLocalFacets());
        for (std::size_t fctNo = 0; fctNo < base::numLocalFacets(); ++fctNo) {
            scratch.reset();
            auto const& info = base::fctInfo[fctNo];
            auto ib0 = info.g_up[0];
            auto ib1 = info.g_up[1];
            if (info.up[0] != info.up[1]) {
                lop_->prepare_skeleton(fctNo, info, scratch);
            } else {
                lop_->prepare_boundary(fctNo, info, scratch);
            }
        }

        for (std::size_t elNo = 0; elNo < base::numElements(); ++elNo) {
            scratch.reset();
            lop_->prepare_volume(elNo, scratch);
        }
        lop_->end_preparation(base::elementScatter_);
    }

    LocalOperator& lop() { return *lop_; }

    template <typename Solver> void assemble(Solver& solver, bool reuse_mat = true) {
        auto bs = lop_->block_size();
        solver.create_mat(bs, base::numLocalElements(),
                          &base::volInfo[0].template get<typename base::NumLocalNeighbours>(),
                          &base::volInfo[0].template get<typename base::NumGhostNeighbours>(),
                          reuse_mat, base::comm());

        auto A00 = Managed<Matrix<double>>(bs, bs);
        auto A01 = Managed<Matrix<double>>(bs, bs);
        auto A10 = Managed<Matrix<double>>(bs, bs);
        auto A11 = Managed<Matrix<double>>(bs, bs);

        auto scratch =
            LinearAllocator(scratch_mem_.get(), scratch_mem_.get() + lop_->scratch_mem_size());
        solver.begin_assembly();
        for (std::size_t elNo = 0; elNo < base::numLocalElements(); ++elNo) {
            scratch.reset();
            if (lop_->assemble_volume(elNo, A00, scratch)) {
                auto ib = base::volInfo[elNo].template get<typename base::GID>();
                solver.add_block(ib, ib, A00);
            }
        }

        for (std::size_t fctNo = 0; fctNo < base::numLocalFacets(); ++fctNo) {
            scratch.reset();
            auto const& info = base::fctInfo[fctNo];
            auto ib0 = info.g_up[0];
            auto ib1 = info.g_up[1];
            if (info.up[0] != info.up[1]) {
                if (lop_->assemble_skeleton(fctNo, info, A00, A01, A10, A11, scratch)) {
                    if (info.inside[0]) {
                        solver.add_block(ib0, ib0, A00);
                        solver.add_block(ib0, ib1, A01);
                    }
                    if (info.inside[1]) {
                        solver.add_block(ib1, ib0, A10);
                        solver.add_block(ib1, ib1, A11);
                    }
                }
            } else {
                if (info.inside[0]) {
                    if (lop_->assemble_boundary(fctNo, info, A00, A01, scratch)) {
                        solver.add_block(ib0, ib0, A00);
                    }
                }
            }
        }
        solver.end_assembly();
    }

    template <typename Solver> void rhs(Solver& solver, bool reuse_vec = true) {
        auto bs = lop_->block_size();
        solver.create_vec(bs, base::numLocalElements(), reuse_vec, base::comm());

        auto B0 = Managed<Vector<double>>(bs);
        auto B1 = Managed<Vector<double>>(bs);

        auto scratch =
            LinearAllocator(scratch_mem_.get(), scratch_mem_.get() + lop_->scratch_mem_size());
        solver.begin_rhs();
        for (std::size_t elNo = 0; elNo < base::numLocalElements(); ++elNo) {
            scratch.reset();
            if (lop_->rhs_volume(elNo, B0, scratch)) {
                auto ib = base::volInfo[elNo].template get<typename base::GID>();
                solver.add_rhs(ib, B0);
            }
        }

        for (std::size_t fctNo = 0; fctNo < base::numLocalFacets(); ++fctNo) {
            scratch.reset();
            auto const& info = base::fctInfo[fctNo];
            auto ib0 = info.g_up[0];
            auto ib1 = info.g_up[1];
            if (info.up[0] != info.up[1]) {
                if (lop_->rhs_skeleton(fctNo, info, B0, B1, scratch)) {
                    if (info.inside[0]) {
                        solver.add_rhs(ib0, B0);
                    }
                    if (info.inside[1]) {
                        solver.add_rhs(ib1, B1);
                    }
                }
            } else {
                if (info.inside[0]) {
                    if (lop_->rhs_boundary(fctNo, info, B0, scratch)) {
                        solver.add_rhs(ib0, B0);
                    }
                }
            }
        }
        solver.end_rhs();
    }

private:
    std::unique_ptr<LocalOperator> lop_;
    std::unique_ptr<char[]> scratch_mem_;
};

} // namespace tndm

#endif // DGOPERATOR_20200909_H
