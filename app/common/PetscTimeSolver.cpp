#include "PetscTimeSolver.h"

namespace tndm {

PetscTimeSolverBase::PetscTimeSolverBase(MPI_Comm comm) {
    CHKERRTHROW(TSCreate(comm, &ts_));
    CHKERRTHROW(TSSetProblemType(ts_, TS_NONLINEAR));
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

std::string PetscTimeSolverBase::get_checkpoint_filename(void) {
    return ts_filename_;
}

void PetscTimeSolverBase::set_checkpoint_filename(std::string fname) {
    ts_filename_ = fname;
}

PetscInt PetscTimeSolverBase::get_checkpoint_frequency(void) {
    return checkpoint_every_nsteps_;
}

void PetscTimeSolverBase::set_checkpoint_frequency(PetscInt nt) {
    checkpoint_every_nsteps_ = nt;
}

} // namespace tndm
