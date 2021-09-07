#ifndef SEASQDOPERATOR_20210906_H
#define SEASQDOPERATOR_20210906_H

#include "common/MGConfig.h"
#include "common/PetscLinearSolver.h"
#include "common/PetscVector.h"
#include "config.h"

#include "form/BoundaryMap.h"
#include "form/FiniteElementFunction.h"
#include "geometry/Curvilinear.h"
#include "interface/BlockVector.h"
#include "parallel/LocalGhostCompositeView.h"
#include "parallel/Scatter.h"
#include "parallel/SparseBlockVector.h"
#include "tensor/Managed.h"
#include "tensor/Reshape.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"
#include "util/Range.h"
#include "util/Scratch.h"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tndm {

template <typename AdapterOperator, typename DomainOperator, typename FrictionOperator>
class SeasQDOperator {
public:
    constexpr static std::size_t Dim = DomainOperator::local_operator_t::Dim;
    constexpr static std::size_t DomainNumQuantities =
        DomainOperator::local_operator_t::NumQuantities;
    using time_functional_t =
        std::function<std::array<double, DomainNumQuantities>(std::array<double, Dim + 1u> const&)>;

    SeasQDOperator(std::unique_ptr<DomainOperator> dgop, std::unique_ptr<AdapterOperator> adapter,
                   std::unique_ptr<FrictionOperator> friction, bool matrix_free = false,
                   MGConfig const& mg_config = MGConfig())
        : dgop_(std::move(dgop)), linear_solver_(*dgop_, matrix_free, mg_config),
          adapter_(std::move(adapter)), friction_(std::move(friction)),
          traction_(adapter_->lop().traction_block_size(), adapter_->num_local_elements(),
                    adapter_->comm()),
          disp_scatter_(dgop_->topo().elementScatterPlan()),
          disp_ghost_(
              disp_scatter_.recv_prototype<double>(dgop_->block_size(), dgop_->lop().alignment())),
          state_scatter_(adapter_->fault_map().scatter_plan()),
          state_ghost_(
              state_scatter_.recv_prototype<double>(friction_->lop().block_size(), ALIGNMENT)) {}

    void warmup() { linear_solver_.warmup(); }

    void set_boundary(time_functional_t fun) { fun_boundary_ = std::move(fun); }

    std::size_t block_size() const { return friction_->block_size(); }
    std::size_t num_local_elements() const { return friction_->num_local_elements(); }
    MPI_Comm comm() const { return dgop_->topo().comm(); }

    AdapterOperator& adapter() { return *adapter_; }
    AdapterOperator const& adapter() const { return *adapter_; }
    DomainOperator& domain() { return *dgop_; }
    DomainOperator const& domain() const { return *dgop_; }
    FrictionOperator& friction() { return *friction_; }
    FrictionOperator const& friction() const { return *friction_; }

    void initial_condition(BlockVector& state) {
        friction_->pre_init(state);

        solve(0.0, state);
        update_traction(state);

        friction_->init(traction_, state);
    }

    void rhs(double time, BlockVector const& state, BlockVector& result) {
        solve(time, state);
        update_traction(state);

        friction_->rhs(time, traction_, state, result);
    }

    void update_internal_state(double time, BlockVector const& state,
                               bool state_changed_since_last_rhs, bool require_traction,
                               bool require_displacement) {
        bool require_solve =
            state_changed_since_last_rhs && (require_traction || require_displacement);
        if (!require_solve) {
            return;
        }

        solve(time, state);
        if (require_traction) {
            update_traction(state);
        }
    }

    template <typename Iterator> auto displacement(Iterator first, Iterator last) const {
        return dgop_->solution(linear_solver_.x(), std::forward<Iterator>(first),
                               std::forward<Iterator>(last));
    }
    auto displacement() const { return dgop_->solution(linear_solver_.x()); }

    template <typename Iterator>
    auto state(BlockVector const& state_vec, Iterator first, Iterator last) {
        return friction_->state(traction_, state_vec, first, last);
    }
    auto state(BlockVector const& state_vec) { return friction_->state(traction_, state_vec); }

private:
    auto invalid_slip_bc() {
        return [](std::size_t, Matrix<double>&, bool) {
            throw std::logic_error("Slip boundary condition not set");
        };
    }

    void solve(double time, BlockVector const& state) {
        state_scatter_.begin_scatter(state, state_ghost_);
        state_scatter_.wait_scatter();
        auto state_view = LocalGhostCompositeView(state, state_ghost_);
        dgop_->lop().set_slip(adapter_->slip_bc(state_view));
        dgop_->lop().set_dirichlet(
            [this, time](std::array<double, Dim> const& x) {
                std::array<double, Dim + 1u> xt;
                std::copy(x.begin(), x.end(), xt.begin());
                xt.back() = time;
                return this->fun_boundary_(xt);
            },
            adapter_->lop().ref_normal());
        linear_solver_.update_rhs(*dgop_);
        linear_solver_.solve();
        dgop_->lop().set_slip(invalid_slip_bc());
        disp_scatter_.begin_scatter(linear_solver_.x(), disp_ghost_);
        disp_scatter_.wait_scatter();
    }

    void update_traction(BlockVector const& state) {
        auto disp_view = LocalGhostCompositeView(linear_solver_.x(), disp_ghost_);
        auto state_view = LocalGhostCompositeView(state, state_ghost_);
        dgop_->lop().set_slip(adapter_->slip_bc(state_view));
        adapter_->traction(*dgop_, disp_view, traction_);
        dgop_->lop().set_slip(invalid_slip_bc());
    }

    std::unique_ptr<DomainOperator> dgop_;
    PetscLinearSolver linear_solver_;
    std::unique_ptr<AdapterOperator> adapter_;
    std::unique_ptr<FrictionOperator> friction_;

    PetscVector traction_;
    Scatter disp_scatter_;
    SparseBlockVector<double> disp_ghost_;

    Scatter state_scatter_;
    SparseBlockVector<double> state_ghost_;

    time_functional_t fun_boundary_ =
        [](std::array<double, Dim + 1u> const& x) -> std::array<double, DomainNumQuantities> {
        return {};
    };
};

} // namespace tndm

#endif // SEASQDOPERATOR_20210906_H
