#include "SeasQDOperator.h"

#include "form/RefElement.h"

namespace tndm {

SeasQDOperator::SeasQDOperator(std::unique_ptr<dg_t> dgop,
                               std::unique_ptr<AbstractAdapterOperator> adapter,
                               std::unique_ptr<AbstractFrictionOperator> friction, bool matrix_free,
                               MGConfig const& mg_config)
    : dgop_(std::move(dgop)), linear_solver_(*dgop_, matrix_free, mg_config),
      adapter_(std::move(adapter)), friction_(std::move(friction)),
      disp_scatter_(dgop_->topo().elementScatterPlan()),
      disp_ghost_(disp_scatter_.recv_prototype<double>(dgop_->block_size(), ALIGNMENT)),
      state_scatter_(adapter_->fault_map().scatter_plan()),
      state_ghost_(state_scatter_.recv_prototype<double>(friction_->block_size(), ALIGNMENT)),
      traction_(adapter_->traction_block_size(), adapter_->num_local_elements(), adapter_->comm()) {
}
void SeasQDOperator::initial_condition(BlockVector& state) {
    friction_->pre_init(state);

    update_ghost_state(state);
    solve(0.0, make_state_view(state));
    update_traction(make_state_view(state));

    friction_->init(0.0, traction_, state);
}

void SeasQDOperator::rhs(double time, BlockVector const& state, BlockVector& result) {
    update_ghost_state(state);
    solve(time, make_state_view(state));
    update_traction(make_state_view(state));

    friction_->rhs(time, traction_, state, result);
}

void SeasQDOperator::update_internal_state(double time, BlockVector const& state,
                                           bool state_changed_since_last_rhs, bool require_traction,
                                           bool require_displacement) {
    bool require_solve = state_changed_since_last_rhs && (require_traction || require_displacement);
    if (!require_solve) {
        return;
    }

    update_ghost_state(state);
    solve(time, make_state_view(state));
    if (require_traction) {
        update_traction(make_state_view(state));
    }
}

void SeasQDOperator::solve(double time, BlockView const& state_view) {
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

void SeasQDOperator::update_traction(BlockView const& state_view) {
    auto disp_view = LocalGhostCompositeView(linear_solver_.x(), disp_ghost_);
    dgop_->set_slip(adapter_->slip_bc(state_view));
    adapter_->traction(disp_view, traction_);
    dgop_->set_slip(invalid_slip_bc());
}

} // namespace tndm
