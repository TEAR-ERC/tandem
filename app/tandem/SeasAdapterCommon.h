#ifndef SEASADAPTERCOMMON_20210413_H
#define SEASADAPTERCOMMON_20210413_H

#include "common/MGConfig.h"
#include "common/PetscLinearSolver.h"
#include "tandem/SeasAdapterBase.h"

#include "form/DGOperator.h"
#include "form/DGOperatorTopo.h"
#include "geometry/Curvilinear.h"
#include "interface/BlockVector.h"
#include "interface/BlockView.h"
#include "parallel/SparseBlockVector.h"
#include "tensor/Tensor.h"
#include "tensor/TensorBase.h"
#include "util/LinearAllocator.h"

#include <array>
#include <cstddef>
#include <memory>
#include <utility>

namespace tndm {

template <typename Derived, typename LocalOperatorT>
class SeasAdapterCommon : public SeasAdapterBase {
public:
    using local_operator_t = LocalOperatorT;
    constexpr static std::size_t Dim = local_operator_t::Dim;
    constexpr static std::size_t NumQuantities = local_operator_t::NumQuantities;
    using time_functional_t =
        std::function<std::array<double, NumQuantities>(std::array<double, Dim + 1u> const&)>;

    SeasAdapterCommon(std::unique_ptr<BoundaryMap> fault_map,
                      std::unique_ptr<RefElement<Dim - 1u>> space,
                      std::unique_ptr<DGOperator<local_operator_t>> dgop,
                      std::array<double, Dim> const& up, std::array<double, Dim> const& ref_normal,
                      bool matrix_free = false, MGConfig const& mg_config = MGConfig())
        : SeasAdapterBase(std::move(fault_map), std::move(dgop->lop().cl_ptr()),
                          std::move(dgop->topo_ptr()), std::move(space),
                          dgop->lop().facetQuadratureRule().points(), up, ref_normal),
          dgop_(std::move(dgop)), linear_solver_(*dgop_, matrix_free, mg_config),
          scatter_(topo_->elementScatterPlan()),
          ghost_(scatter_.recv_prototype<double>(dgop_->block_size(), dgop_->lop().alignment())) {}

    void set_boundary(time_functional_t fun) { fun_boundary = std::move(fun); }

    void warmup() { linear_solver_.warmup(); }

    void solve(double time, BlockView& state) {
        dgop_->lop().set_slip([this, &state](std::size_t fctNo, Matrix<double>& f_q, bool) {
            auto faultNo = this->faultMap_->bndNo(fctNo);
            auto state_block = state.get_block(faultNo);
            static_cast<Derived*>(this)->slip(faultNo, state_block, f_q);
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
        scatter_.begin_scatter(linear_solver_.x(), ghost_);
        scatter_.wait_scatter();
    }

    void begin_traction(BlockView& state) {
        handle_ = linear_solver_.x().begin_access_readonly();
        dgop_->lop().set_slip([this, &state](std::size_t fctNo, Matrix<double>& f_q, bool) {
            auto faultNo = this->faultMap_->bndNo(fctNo);
            auto state_block = state.get_block(faultNo);
            static_cast<Derived*>(this)->slip(faultNo, state_block, f_q);
        });
    }
    void traction(std::size_t faultNo, Matrix<double>& traction,
                  LinearAllocator<double>& scratch) const {
        static_cast<Derived const*>(this)->traction(faultNo, traction, scratch);
    }
    void end_traction() { linear_solver_.x().end_access_readonly(handle_); }

    template <typename Iterator> auto displacement(Iterator first, Iterator last) const {
        return dgop_->solution(linear_solver_.x(), std::forward<Iterator>(first),
                               std::forward<Iterator>(last));
    }
    auto displacement() const { return dgop_->solution(linear_solver_.x()); }
    std::size_t numLocalElements() const { return dgop_->numLocalElements(); }
    std::size_t number_of_local_dofs() const { return dgop_->number_of_local_dofs(); }

protected:
    std::unique_ptr<DGOperator<local_operator_t>> dgop_;
    PetscLinearSolver linear_solver_;
    Scatter scatter_;
    SparseBlockVector<double> ghost_;

    time_functional_t fun_boundary =
        [](std::array<double, Dim + 1u> const& x) -> std::array<double, NumQuantities> {
        return {};
    };
    Matrix<const double> handle_;
};

} // namespace tndm

#endif // SEASADAPTERCOMMON_20210413_H
