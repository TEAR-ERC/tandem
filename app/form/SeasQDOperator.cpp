#include "SeasQDOperator.h"

#include "common/PetscUtil.h"
#include "form/RefElement.h"

#include <petscsys.h>
#include <petscvec.h>

#include <algorithm>
#include <vector>

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

SeasQDOperator::BenchResult SeasQDOperator::benchmark_solve(int nreps) {
    BenchResult result;
    MPI_Comm comm = this->comm();
    MPI_Comm_size(comm, &result.n_ranks);
    result.n_reps = nreps;

    // Build a random slip state vector and scatter its ghost elements once.
    // solve() only reads from the state view, so a single scatter is enough.
    auto slip_bs = friction_->block_size();
    auto n_local = friction_->num_local_elements();
    PetscVector slip_vec(slip_bs, n_local, comm);
    {
        PetscRandom rng;
        CHKERRTHROW(PetscRandomCreate(comm, &rng));
        CHKERRTHROW(PetscRandomSetType(rng, PETSCRAND48));
        CHKERRTHROW(PetscRandomSetSeed(rng, 11223344UL));
        CHKERRTHROW(PetscRandomSeed(rng));
        CHKERRTHROW(VecSetRandom(slip_vec.vec(), rng));
        CHKERRTHROW(PetscRandomDestroy(&rng));
    }
    slip_vec.begin_assembly();
    slip_vec.end_assembly();

    state_scatter_.begin_scatter(slip_vec, state_ghost_);
    state_scatter_.wait_scatter();
    auto state_view = LocalGhostCompositeView(slip_vec, state_ghost_);

    // Warmup
    solve(0.0, state_view);
    update_traction(state_view);

    // Timed reps
    std::vector<double> times(nreps);
    for (int r = 0; r < nreps; ++r) {
        PetscLogDouble t0, t1;
        CHKERRTHROW(MPI_Barrier(comm));
        CHKERRTHROW(PetscTime(&t0));
        solve(0.0, state_view);
        update_traction(state_view);
        CHKERRTHROW(PetscTime(&t1));
        times[r] = t1 - t0;
    }

    // Wall-clock = max across ranks per rep
    std::vector<double> wall(nreps);
    CHKERRTHROW(MPI_Allreduce(times.data(), wall.data(), nreps, MPI_DOUBLE, MPI_MAX, comm));

    double sum = 0.0, mn = wall[0], mx = wall[0];
    for (double t : wall) {
        sum += t;
        mn = std::min(mn, t);
        mx = std::max(mx, t);
    }
    result.time_avg_s = sum / nreps;
    result.time_min_s = mn;
    result.time_max_s = mx;
    return result;
}

} // namespace tndm
