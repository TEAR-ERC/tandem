#ifndef SEASQDOPERATOR_20210906_H
#define SEASQDOPERATOR_20210906_H

#include "common/MGConfig.h"
#include "common/PetscLinearSolver.h"
#include "common/PetscVector.h"
#include "config.h"
#include "form/AbstractAdapterOperator.h"
#include "form/AbstractFrictionOperator.h"
#include "form/FacetFunctionalFactory.h"

#include "form/AbstractDGOperator.h"
#include "form/BoundaryMap.h"
#include "interface/BlockVector.h"
#include "parallel/LocalGhostCompositeView.h"
#include "parallel/Scatter.h"
#include "parallel/SparseBlockVector.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"
#include "util/Scratch.h"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tndm {

class SeasQDOperator {
public:
    using dg_t = AbstractDGOperator<DomainDimension>;

    SeasQDOperator(std::unique_ptr<dg_t> dgop, std::unique_ptr<AbstractAdapterOperator> adapter,
                   std::unique_ptr<AbstractFrictionOperator> friction, bool matrix_free = false,
                   MGConfig const& mg_config = MGConfig())
        : dgop_(std::move(dgop)), linear_solver_(*dgop_, matrix_free, mg_config),
          adapter_(std::move(adapter)), friction_(std::move(friction)),
          disp_scatter_(dgop_->topo().elementScatterPlan()),
          disp_ghost_(disp_scatter_.recv_prototype<double>(dgop_->block_size(), ALIGNMENT)),
          state_scatter_(adapter_->fault_map().scatter_plan()),
          state_ghost_(state_scatter_.recv_prototype<double>(friction_->block_size(), ALIGNMENT)),
          traction_(adapter_->traction_block_size(), adapter_->num_local_elements(),
                    adapter_->comm()) {}

    void warmup() { linear_solver_.warmup(); }

    void set_boundary(std::unique_ptr<AbstractFacetFunctionalFactory> fun) {
        fun_boundary_ = std::move(fun);
    }

    auto block_sizes() -> std::array<std::size_t, 1> const { return {friction_->block_size()}; }
    auto num_local_elements() -> std::array<std::size_t, 1> const {
        return {friction_->num_local_elements()};
    }
    MPI_Comm comm() const { return dgop_->topo().comm(); }

    AbstractAdapterOperator& adapter() { return *adapter_; }
    AbstractAdapterOperator const& adapter() const { return *adapter_; }
    dg_t& domain() { return *dgop_; }
    dg_t const& domain() const { return *dgop_; }
    AbstractFrictionOperator& friction() { return *friction_; }
    AbstractFrictionOperator const& friction() const { return *friction_; }

    void initial_condition(BlockVector& state) {
        friction_->pre_init(state);

        update_ghost_state(state);
        solve(0.0, make_state_view(state));
        update_traction(make_state_view(state));

        friction_->init(traction_, state);
    }

    void rhs(double time, BlockVector const& state, BlockVector& result) {
        update_ghost_state(state);
        solve(time, make_state_view(state));
        update_traction(make_state_view(state));

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

        update_ghost_state(state);
        solve(time, make_state_view(state));
        if (require_traction) {
            update_traction(make_state_view(state));
        }
    }

    auto displacement(std::vector<std::size_t> const& subset) const {
        return dgop_->solution(linear_solver_.x(), subset);
    }
    auto displacement() const { return dgop_->solution(linear_solver_.x()); }

    auto state(BlockVector const& state_vec, std::vector<std::size_t> const& subset) {
        return friction_->state(traction_, state_vec, subset);
    }
    auto state(BlockVector const& state_vec) { return friction_->state(traction_, state_vec); }

protected:
    auto invalid_slip_bc() {
        return [](std::size_t, Matrix<double>&, bool) {
            throw std::logic_error("Slip boundary condition not set");
        };
    }

    void update_ghost_state(BlockVector const& state) {
        state_scatter_.begin_scatter(state, state_ghost_);
        state_scatter_.wait_scatter();
    }

    auto make_state_view(BlockVector const& state) -> LocalGhostCompositeView {
        return LocalGhostCompositeView(state, state_ghost_);
    }

    void solve(double time, BlockView const& state_view) {
        dgop_->set_slip(adapter_->slip_bc(state_view));
        if (fun_boundary_) {
            dgop_->set_dirichlet((*fun_boundary_)(time));
        }
        linear_solver_.update_rhs(*dgop_);
        linear_solver_.solve();
        dgop_->set_slip(invalid_slip_bc());
        disp_scatter_.begin_scatter(linear_solver_.x(), disp_ghost_);
        disp_scatter_.wait_scatter();
    }

    void update_traction(BlockView const& state_view) {
        auto disp_view = LocalGhostCompositeView(linear_solver_.x(), disp_ghost_);
        dgop_->set_slip(adapter_->slip_bc(state_view));
        adapter_->traction(disp_view, traction_);
        dgop_->set_slip(invalid_slip_bc());
    }

private:
    std::unique_ptr<dg_t> dgop_;
    PetscLinearSolver linear_solver_;
    std::unique_ptr<AbstractAdapterOperator> adapter_;
    std::unique_ptr<AbstractFrictionOperator> friction_;

    Scatter disp_scatter_;
    SparseBlockVector<double> disp_ghost_;

    Scatter state_scatter_;
    SparseBlockVector<double> state_ghost_;

    std::unique_ptr<AbstractFacetFunctionalFactory> fun_boundary_ = nullptr;

protected:
    PetscVector traction_;
};

} // namespace tndm

#endif // SEASQDOPERATOR_20210906_H
