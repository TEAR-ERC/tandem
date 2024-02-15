#include "PetscTimeSolver.h"

namespace tndm {

PetscTimeSolverBase::PetscTimeSolverBase(MPI_Comm comm, Config const& cfg) {
    CHKERRTHROW(TSCreate(comm, &ts_));
    CHKERRTHROW(TSSetProblemType(ts_, TS_NONLINEAR));
    CHKERRTHROW(TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP));
    CHKERRTHROW(TSSetFromOptions(ts_));

    auto const& cfgcp = cfg.ts_checkpoint_config;

    CHKERRTHROW(
        ts_checkpoint_configure(ts_, cfgcp.save_directory.c_str(), cfgcp.frequency_step,
                                cfgcp.frequency_cputime_minutes, cfgcp.frequency_time_physical,
                                static_cast<int>(cfgcp.storage_type), cfgcp.storage_limited_size));

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

PetscTimeSolverBase::~PetscTimeSolverBase() { TSDestroy(&ts_); }

std::size_t PetscTimeSolverBase::get_step_number() const {
    PetscInt steps;
    CHKERRTHROW(TSGetStepNumber(ts_, &steps));
    return steps;
}

std::size_t PetscTimeSolverBase::get_step_rejections() const {
    PetscInt rejects;
    CHKERRTHROW(TSGetStepRejections(ts_, &rejects));
    return rejects;
}

void PetscTimeSolverBase::set_max_time_step(double dt) {
    TSAdapt adapt;
    CHKERRTHROW(TSGetAdapt(ts_, &adapt));
    CHKERRTHROW(TSAdaptSetStepLimits(adapt, PETSC_DEFAULT, dt));
    CHKERRTHROW(TSSetTimeStep(ts_, dt));
}

} // namespace tndm
