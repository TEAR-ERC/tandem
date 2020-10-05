#ifndef PETSCTS_20201001_H
#define PETSCTS_20201001_H

#include "common/PetscBlockVector.h"
#include "common/PetscUtil.h"
#include "tensor/Tensor.h"

#include <petscsys.h>
#include <petscsystypes.h>
#include <petscts.h>

#include <cstddef>
#include <memory>

namespace tndm {

class PetscTimeSolver {
public:
    template <typename TimeOp> PetscTimeSolver(TimeOp& timeop) {
        state_ = std::make_unique<PetscBlockVector>(timeop.block_size(), timeop.numLocalElements(),
                                                    timeop.comm());
        timeop.initial_condition(*state_);

        CHKERRTHROW(TSCreate(timeop.comm(), &ts_));
        CHKERRTHROW(TSSetProblemType(ts_, TS_NONLINEAR));
        CHKERRTHROW(TSSetSolution(ts_, state_->vec()));
        CHKERRTHROW(TSSetRHSFunction(ts_, nullptr, RHSFunction<TimeOp>, &timeop));
        CHKERRTHROW(TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP));
        CHKERRTHROW(TSSetFromOptions(ts_));
    }
    ~PetscTimeSolver() { TSDestroy(&ts_); }

    void solve(double upcoming_time) {
        CHKERRTHROW(TSSetMaxTime(ts_, upcoming_time));
        CHKERRTHROW(TSSolve(ts_, state_->vec()));
    }

    auto& state() { return *state_; }
    auto const& state() const { return *state_; }

private:
    template <typename TimeOp>
    static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec u, Vec F, void* ctx) {
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);
        auto u_view = PetscBlockVectorView(u);
        auto F_view = PetscBlockVectorView(F);
        self->rhs(u_view, F_view);
        return 0;
    }

    std::unique_ptr<PetscBlockVector> state_;
    TS ts_ = nullptr;
};

} // namespace tndm

#endif // PETSCTS_20201001_H
