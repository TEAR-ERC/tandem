#ifndef SEASOPERATOR_20201001_H
#define SEASOPERATOR_20201001_H

#include "common/CmdLine.h"
#include "common/PetscLinearSolver.h"

#include "form/BC.h"
#include "form/BoundaryMap.h"
#include "form/DGOperatorTopo.h"
#include "tensor/Managed.h"
#include "tensor/Reshape.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tndm {

template <typename LocalOperator, typename SpatialOperator> class SeasOperator {
public:
    constexpr static std::size_t Dim = SpatialOperator::local_operator_t::Dim;
    constexpr static std::size_t NumQuantities = SpatialOperator::local_operator_t::NumQuantities;
    using time_functional_t =
        std::function<std::array<double, NumQuantities>(std::array<double, Dim + 1u> const&)>;

    SeasOperator(std::shared_ptr<DGOperatorTopo> topo, std::unique_ptr<LocalOperator> localOperator,
                 std::unique_ptr<SpatialOperator> spatialOperator,
                 std::array<double, Dim> const& ref_normal)
        : topo_(std::move(topo)), lop_(std::move(localOperator)), dgop_(std::move(spatialOperator)),
          faultMap_(*topo_, BC::Fault), linear_solver_(*dgop_), ref_normal_(ref_normal) {
        std::size_t numFacets = topo_->numLocalFacets();
        scratch_mem_ = std::make_unique<char[]>(lop_->scratch_mem_size());
        auto scratch = make_scratch();
        lop_->begin_preparation(numLocalElements());
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto fctNo = faultMap_.fctNo(faultNo);
            lop_->prepare(faultNo, topo_->info(fctNo), ref_normal_, scratch);
        }
        lop_->end_preparation();
    }

    std::size_t block_size() const { return lop_->block_size(); }
    std::size_t numLocalElements() const { return faultMap_.size(); }
    MPI_Comm comm() const { return topo_->comm(); }
    BoundaryMap const& faultMap() const { return faultMap_; }
    SpatialOperator const& spatialOperator() const { return *dgop_; }
    auto displacement() const { return dgop_->solution(linear_solver_.x()); }

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

    template <typename BlockVector> void rhs(double time, BlockVector& state, BlockVector& result) {
        auto scratch = make_scratch();
        auto in_handle = state.begin_access_readonly();
        dgop_->lop().set_slip(
            [this, &state, &in_handle, &scratch](std::size_t fctNo, Matrix<double>& f_q, bool) {
                auto faultNo = this->faultMap_.bndNo(fctNo);
                auto state_block = state.get_block(in_handle, faultNo);
                this->lop_->slip(faultNo, state_block, f_q, scratch);
            });
        dgop_->lop().set_dirichlet(
            [this, time](std::array<double, Dim> const& x) {
                std::array<double, Dim + 1u> xt;
                std::copy(x.begin(), x.end(), xt.begin());
                xt.back() = time;
                return this->fun_boundary(xt);
            },
            ref_normal_);
        linear_solver_.update_rhs(*dgop_);
        linear_solver_.solve();

        auto traction = Managed<Matrix<double>>(dgop_->lop().tractionResultInfo());
        auto const& coupled_dofs = linear_solver_.x();
        auto out_handle = result.begin_access();
        auto coupled_handle = coupled_dofs.begin_access_readonly();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto fctNo = faultMap_.fctNo(faultNo);
            auto const& info = topo_->info(fctNo);
            auto u0 = coupled_dofs.get_block(coupled_handle, info.up[0]);
            auto u1 = coupled_dofs.get_block(coupled_handle, info.up[1]);
            dgop_->lop().traction(fctNo, info, u0, u1, traction);

            auto state_block = state.get_block(in_handle, faultNo);
            auto result_block = result.get_block(out_handle, faultNo);
            double VMax = lop_->rhs(faultNo, traction, state_block, result_block, scratch);
            VMax_ = std::max(VMax_, VMax);
        }
        coupled_dofs.end_access_readonly(coupled_handle);
        state.end_access_readonly(in_handle);
        result.end_access(out_handle);
    }

    template <typename BlockVector> auto state(BlockVector& vector) {
        auto soln = lop_->state_prototype(numLocalElements());
        auto& values = soln.values();

        auto traction = Managed<Matrix<double>>(dgop_->lop().tractionResultInfo());
        auto const& coupled_dofs = linear_solver_.x();
        auto coupled_handle = coupled_dofs.begin_access_readonly();
        auto in_handle = vector.begin_access_readonly();
        auto scratch = make_scratch();
        for (std::size_t faultNo = 0, num = numLocalElements(); faultNo < num; ++faultNo) {
            auto fctNo = faultMap_.fctNo(faultNo);
            auto const& info = topo_->info(fctNo);
            auto u0 = coupled_dofs.get_block(coupled_handle, info.up[0]);
            auto u1 = coupled_dofs.get_block(coupled_handle, info.up[1]);
            dgop_->lop().traction(fctNo, info, u0, u1, traction);

            auto value_matrix = values.subtensor(slice{}, slice{}, faultNo);
            auto state_block = vector.get_block(in_handle, faultNo);
            lop_->state(faultNo, traction, state_block, value_matrix, scratch);
        }
        vector.end_access_readonly(in_handle);
        coupled_dofs.end_access_readonly(coupled_handle);
        return soln;
    }

    void set_boundary(time_functional_t fun) { fun_boundary = std::move(fun); }

    double VMax() const { return VMax_; }

private:
    auto make_scratch() const {
        return LinearAllocator(scratch_mem_.get(), scratch_mem_.get() + lop_->scratch_mem_size());
    }
    std::shared_ptr<DGOperatorTopo> topo_;
    std::unique_ptr<LocalOperator> lop_;
    std::unique_ptr<SpatialOperator> dgop_;
    std::unique_ptr<char[]> scratch_mem_;
    BoundaryMap faultMap_;
    PetscLinearSolver linear_solver_;
    time_functional_t fun_boundary =
        [](std::array<double, Dim + 1u> const& x) -> std::array<double, NumQuantities> {
        return {};
    };
    std::array<double, Dim> ref_normal_;
    double VMax_ = 0.0;
};

} // namespace tndm

#endif // SEASOPERATOR_20201001_H
