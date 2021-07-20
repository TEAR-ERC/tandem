#ifndef PETSCTS_20201001_H
#define PETSCTS_20201001_H

#include "common/PetscUtil.h"
#include "common/PetscVector.h"

#include <petscsystypes.h>
#include <petscts.h>
#include <petscvec.h>

#include <memory>

namespace tndm {

class PetscTimeSolver {
public:
    template <typename TimeOp> PetscTimeSolver(TimeOp& timeop) {
        state_ = std::make_unique<PetscVector>(timeop.block_size(), timeop.numLocalElements(),
                                               timeop.comm());
        timeop.initial_condition(*state_);

        CHKERRTHROW(TSCreate(timeop.comm(), &ts_));
        CHKERRTHROW(TSSetProblemType(ts_, TS_NONLINEAR));
        CHKERRTHROW(TSSetSolution(ts_, state_->vec()));
        CHKERRTHROW(TSSetRHSFunction(ts_, nullptr, RHSFunction<TimeOp>, &timeop));
        CHKERRTHROW(TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP));
        CHKERRTHROW(TSSetFromOptions(ts_));

        TSType time_scheme;
        CHKERRTHROW(TSGetType(ts_, &time_scheme));

        // Check whether time integrator has First Same As Last (FSAL) property
        switch (fnv1a(time_scheme)) {
        case HASH_DEF(TSRK): {
            PetscBool FSAL;
            CHKERRTHROW(TSRKGetTableau(ts_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                       nullptr, &FSAL));
            fsal_ = FSAL == PETSC_TRUE;
            break;
        }
        default:
            fsal_ = false;
            break;
        };
    }
    ~PetscTimeSolver() { TSDestroy(&ts_); }

    void solve(double upcoming_time) {
        CHKERRTHROW(TSSetMaxTime(ts_, upcoming_time));
        CHKERRTHROW(TSSolve(ts_, state_->vec()));
    }

    std::size_t get_step_number() const {
        PetscInt steps;
        CHKERRTHROW(TSGetStepNumber(ts_, &steps));
        return steps;
    }

    std::size_t get_step_rejections() const {
        PetscInt rejects;
        CHKERRTHROW(TSGetStepRejections(ts_, &rejects));
        return rejects;
    }

    auto& state() { return *state_; }
    auto const& state() const { return *state_; }

    template <class Monitor> void set_monitor(Monitor& monitor) {
        CHKERRTHROW(TSMonitorSet(ts_, &MonitorFunction<Monitor>, &monitor, nullptr));
    }

    bool fsal() const { return fsal_; }

private:
    template <typename TimeOp>
    static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec u, Vec F, void* ctx) {
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);
        auto u_view = PetscVectorView(u);
        auto F_view = PetscVectorView(F);
        self->rhs(t, u_view, F_view);
        return 0;
    }

    template <class Monitor>
    static PetscErrorCode MonitorFunction(TS ts, PetscInt steps, PetscReal time, Vec u, void* ctx) {
        Monitor* self = reinterpret_cast<Monitor*>(ctx);
        auto u_view = PetscVectorView(u);
        self->monitor(time, u_view);
        return 0;
    }

    std::unique_ptr<PetscVector> state_;
    TS ts_ = nullptr;
    bool fsal_;
};

} // namespace tndm

#endif // PETSCTS_20201001_H
