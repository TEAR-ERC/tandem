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

void SeasQDOperator::set_boundary(std::unique_ptr<AbstractFacetFunctionalFactory> fun) {
    fun_boundary_ = std::move(fun);
}
void SeasQDOperator::set_free_slip_boundary(std::unique_ptr<AbstractFacetFunctionalFactory> fun) {
    fun_free_slip_boundary_ = std::move(fun);
}
void SeasQDOperator::set_traction_boundary(std::unique_ptr<AbstractFacetFunctionalFactory> fun) {
    fun_traction_boundary_ = std::move(fun);
}

void SeasQDOperator::prepare_for_dt(double dt) {
    // Recompute dt-dependent coefficients; reassemble the matrix only if they
    // changed (viscoelasticity with a fault). No-op for elasticity and for
    // viscoelasticity without a fault.
    if (dgop_->update_time_step(dt)) {
        linear_solver_.reassemble(*dgop_);
        // The matrix changed, so whatever x() holds is no longer its solution.
        solution_valid_ = false;
    }
}

void SeasQDOperator::initial_condition(BlockVector& state) {
    friction_->pre_init(state);

    update_ghost_state(state);
    // For viscoelastic operators, initialize time-dependent coefficients to match the
    // matrix assembled at construction (dt = theta * tau, the cap). For non-viscoelastic
    // operators relaxation_time_global() is 0 and the value passed is ignored.
    double const tau = dgop_->relaxation_time_global();
    double const init_dt = (tau > 0.0) ? tau * dgop_->viscoelastic_theta() : 1e-12;
    prepare_for_dt(init_dt);
    pre_step_update_strain_history();

    solve(0.0, make_state_view(state));
    update_traction(make_state_view(state));
    // Instantaneous elastic initial condition for the viscoelastic memory variable.
    // A strain present at t=0 fully loads the Maxwell spring (the dashpot cannot move
    // instantaneously), so q(0) = eps(0). Running the standard partial-strain recurrence
    // here would instead give q(0) = g(dt)*eps(0), under-predicting the t=0 stress by
    // mu1*(1-g)*eps(0) = O(dt) and degrading a held-step relaxation test to first order.
    update_ghost_state(state);
    solve(0.0, make_state_view(state));
    dgop_->store_displacement_field(linear_solver_.x());
    dgop_->compute_deviatoric_strain();
    dgop_->initialize_partial_strain();
    last_time_ = 0.0;
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

    // For stress output: store displacement and compute current strain state
    // This is needed before stress_volume() can read the correct values.
    // Note: The PostStep callback will also call update_strain_history() which
    // does similar work, but the Monitor callback runs first.
    if (require_displacement) {
        // Update time-dependent coefficients (g_dt, ratio) for partial strain computation
        double dt = time - last_time_;
        if (dt > 0.0) {
            prepare_for_dt(dt);
        }
    }
}

void SeasQDOperator::pre_step_update_strain_history() {
    // Pre-step: rotate history buffers so RHS reads consistent "_old" values.
    dgop_->update_deviatoric_strain();
    dgop_->update_partial_strain();
}

void SeasQDOperator::post_step_compute_strain_history(double time, BlockVector const& state) {
    // Update viscoelastic strain history after an accepted timestep.
    // For non-VE operators, these are no-ops.

    // The post-step sequence is:
    //   1. Compute dt and update time-dependent coefficients (g_dt, ratio)
    //   2. store_displacement_field: save current displacement for next compute
    //   3. compute_deviatoric_strain: compute new deviatoric strain from displacement
    //   4. compute_partial_strain: integrate partial strain (VE memory variable)
    // Rotation old <= new is handled in pre_step_update_strain_history() before the next RHS call.

    // Ensure displacement corresponds to the accepted TS solution at this time,
    // independent of monitor output frequency.
    //
    // Under an FSAL tableau this solve is redundant: the last row of A equals b, so
    // the final stage is Y[s-1] = y_n + h sum_j b[j] YdotRHS[j] -- the step's own
    // solution -- evaluated at t + h*c[s-1] = t + h. rhs() therefore already solved
    // at exactly this state and x() still holds that displacement. TSPostStep runs
    // before TSMonitor, and nothing between the final stage and here touches the
    // operator (post_step's own prepare_for_dt uses the same dt the stages did), so
    // the cached solve is still the current one.
    //
    // Y[s-1] and the vector TSEvaluateStep builds are equal in exact arithmetic but
    // summed in a different order (VecMAXPY chunks by nv & 0x3, so s-1 and s terms
    // group differently), hence they can differ in the last ulp. Reusing the stage
    // solve is therefore a round-off-level change, not a bit-identical one.
    //
    // Without FSAL, Y[s-1] != y_{n+1} and the solve here is mandatory.
    if (!solution_is_current(time)) {
        update_ghost_state(state);
        solve(time, make_state_view(state));
    }

    // Calculate and set the actual dt for this timestep
    double dt = time - last_time_;
    if (dt < 0.0)
        dt = 0.0;
    if (dt > 0.0) {
        prepare_for_dt(dt);
    }
    last_time_ = time;

    dgop_->store_displacement_field(linear_solver_.x());
    dgop_->compute_deviatoric_strain();
    dgop_->compute_partial_strain();
}

void SeasQDOperator::solve(double time, BlockView const& state_view) {
    dgop_->set_slip(adapter_->slip_bc(state_view));
    if (fun_boundary_) {
        dgop_->set_dirichlet((*fun_boundary_)(time));
    }
    if (fun_traction_boundary_) {
        dgop_->set_traction_boundary((*fun_traction_boundary_)(time));
    }
    if (fun_free_slip_boundary_) {
        dgop_->set_free_slip_boundary((*fun_free_slip_boundary_)(time));
    }
    linear_solver_.update_rhs(*dgop_);
    linear_solver_.solve();
    dgop_->set_slip(invalid_slip_bc());
    disp_scatter_.begin_scatter(linear_solver_.x(), disp_ghost_);
    disp_scatter_.wait_scatter();
    solve_time_ = time;
    solution_valid_ = true;
}

void SeasQDOperator::update_traction(BlockView const& state_view) {
    auto disp_view = LocalGhostCompositeView(linear_solver_.x(), disp_ghost_);
    dgop_->set_slip(adapter_->slip_bc(state_view));
    adapter_->traction(disp_view, traction_);
    dgop_->set_slip(invalid_slip_bc());
}

} // namespace tndm
