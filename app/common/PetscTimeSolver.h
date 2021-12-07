#ifndef PETSCTS_20201001_H
#define PETSCTS_20201001_H

#include "common/PetscUtil.h"
#include "common/PetscVector.h"
extern "C" {
  #include "ts_util.h"
  #include "vecnest_util.h"
}
#include <petscsystypes.h>
#include <petscts.h>
#include <petscvec.h>

#include <array>
#include <cassert>
#include <memory>
#include <tuple>

namespace tndm {

class PetscTimeSolverBase {
public:
    PetscTimeSolverBase(MPI_Comm comm);
    ~PetscTimeSolverBase();

    std::size_t get_step_number() const;
    std::size_t get_step_rejections() const;
    inline bool fsal() const { return fsal_; }
    void set_max_time_step(double dt);
    std::string get_checkpoint_filename(void);
    void set_checkpoint_filename(std::string);
    PetscInt get_checkpoint_frequency(void);
    void set_checkpoint_frequency(PetscInt);

protected:
    TS ts_ = nullptr;
    bool fsal_;
    std::string ts_filename_ = "ts_checkpoint.bin";
    PetscInt checkpoint_every_nsteps_ = 10;
};

template <std::size_t NumStateVecs> class PetscTimeSolver : public PetscTimeSolverBase {
public:
    template <typename TimeOp>
    PetscTimeSolver(TimeOp& timeop, std::array<std::unique_ptr<PetscVector>, NumStateVecs> state)
        : PetscTimeSolverBase(timeop.comm()), state_(std::move(state)) {

        Vec x[NumStateVecs];
        for (std::size_t n = 0; n < NumStateVecs; ++n) {
            x[n] = state_[n]->vec();
        }
        MPI_Comm comm;
        CHKERRTHROW(VecCreateNest(timeop.comm(), NumStateVecs, nullptr, x, &ts_state_));
        CHKERRTHROW(VecNestUpgradeOperations(ts_state_));

        std::apply([&timeop](auto&... x) { timeop.initial_condition((*x)...); }, state_);

        CHKERRTHROW(TSSetSolution(ts_, ts_state_));
        CHKERRTHROW(TSSetRHSFunction(ts_, nullptr, RHSFunction<TimeOp>, &timeop));
    }
    ~PetscTimeSolver() { VecDestroy(&ts_state_); }

    void solve(double upcoming_time) {
        CHKERRTHROW(TSSetMaxTime(ts_, upcoming_time));
        CHKERRTHROW(TSSolve(ts_, ts_state_));
    }

    void solve_with_checkpoints(double upcoming_time) {
      PetscInt ns0,ns = checkpoint_every_nsteps_;
      TSConvergedReason reason;
      PetscInt curr,span;

      // Load an existing checkpoint if it exists
      CHKERRTHROW(tandem_TSLoad(ts_,ts_filename_.c_str()));
      CHKERRTHROW(TSSetMaxTime(ts_,upcoming_time));

      CHKERRTHROW(TSGetMaxSteps(ts_,&ns0));
      CHKERRTHROW(TSGetStepNumber(ts_,&curr));
      CHKERRTHROW(TSSetMaxSteps(ts_,std::min(curr + ns,ns0)));

      reason = TS_CONVERGED_ITERATING;
      while (reason != TS_CONVERGED_TIME) {
        CHKERRTHROW(TSGetStepNumber(ts_,&curr));
        CHKERRTHROW(TSGetMaxSteps(ts_,&span));
        printf("[checkpoint phase] executing steps %d to %d\n",(int)curr,(int)span);

        CHKERRTHROW(TSSolve(ts_, ts_state_));
        CHKERRTHROW(TSGetConvergedReason(ts_,&reason));

        CHKERRTHROW(TSGetStepNumber(ts_,&curr));
        if (curr == ns0) {
          // Checkpoint final trajectory
          CHKERRTHROW(tandem_TSView(ts_,ts_filename_.c_str()));
          break;
        }

        if (reason != TS_CONVERGED_TIME) {
          // Checkpoint
          CHKERRTHROW(tandem_TSView(ts_,ts_filename_.c_str()));
          // Change configuration for next checkpoint cycle
          CHKERRTHROW(TSSetMaxSteps(ts_,std::min(curr + ns,ns0)));
        } else {
          // Checkpoint final trajectory
          CHKERRTHROW(tandem_TSView(ts_,ts_filename_.c_str()));
        }
      }
    }

    auto& state(std::size_t idx) {
        assert(idx < NumStateVecs);
        return *state_[idx];
    }
    auto const& state(std::size_t idx) const {
        assert(idx < NumStateVecs);
        return *state_[idx];
    }

    template <class Monitor> void set_monitor(Monitor& monitor) {
        CHKERRTHROW(TSMonitorSet(ts_, &MonitorFunction<Monitor>, &monitor, nullptr));
    }

private:
    template <typename TimeOp>
    static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec u, Vec F, void* ctx) {
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);

        std::array<Vec, 2 * NumStateVecs> x;
        for (std::size_t n = 0; n < NumStateVecs; ++n) {
            CHKERRTHROW(VecNestGetSubVec(u, n, &x[n]));
            CHKERRTHROW(VecNestGetSubVec(F, n, &x[NumStateVecs + n]));
        }
        auto x_view = std::apply(
            [](auto&... x) -> std::array<PetscVectorView, 2 * NumStateVecs> {
                return {PetscVectorView(x)...};
            },
            x);

        std::apply([&self, &t](auto&... xv) { self->rhs(t, xv...); }, x_view);
        return 0;
    }

    template <class Monitor>
    static PetscErrorCode MonitorFunction(TS ts, PetscInt steps, PetscReal time, Vec u, void* ctx) {
        Monitor* self = reinterpret_cast<Monitor*>(ctx);

        std::array<Vec, NumStateVecs> x;
        for (std::size_t n = 0; n < NumStateVecs; ++n) {
            CHKERRTHROW(VecNestGetSubVec(u, n, &x[n]));
        }
        auto x_view = std::apply(
            [](auto&... x) -> std::array<PetscVectorView, NumStateVecs> {
                return {PetscVectorView(x)...};
            },
            x);

        std::apply([&self, &time](auto&... xv) { self->monitor(time, xv...); }, x_view);
        return 0;
    }


    std::array<std::unique_ptr<PetscVector>, NumStateVecs> state_;
    Vec ts_state_ = nullptr;
};

} // namespace tndm

#endif // PETSCTS_20201001_H
