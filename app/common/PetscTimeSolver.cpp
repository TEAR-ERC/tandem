#include "PetscTimeSolver.h"

namespace tndm {

PetscTimeSolverBase::PetscTimeSolverBase(MPI_Comm comm, Config const& cfg) {
    CHKERRTHROW(TSCreate(comm, &ts_));
    CHKERRTHROW(TSSetProblemType(ts_, TS_NONLINEAR));
    CHKERRTHROW(TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP));
    CHKERRTHROW(TSSetFromOptions(ts_));

    // Extract individual members from the C++ struct
    auto const& tsCheckpointConfig = cfg.ts_checkpoint_config;
    const char* saveDirectory = tsCheckpointConfig.save_directory.c_str();
    // tsCheckpointStorageType storageType  = static_cast<enum
    // tndm::tsCheckpointStorageType>(tsCheckpointConfig.storage_type);
    int storageType = static_cast<int>(tsCheckpointConfig.storage_type);

    CHKERRTHROW(ts_checkpoint_configure(ts_, saveDirectory, tsCheckpointConfig.frequency_step,
                                        tsCheckpointConfig.frequency_cputime_minutes,
                                        tsCheckpointConfig.frequency_time_physical, storageType,
                                        tsCheckpointConfig.storage_limited_size));

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
