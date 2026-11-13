#include "PetscTimeSolver.h"
#include "common/PetscLoggingUtils.h"

namespace tndm {

PetscTimeSolverBase::PetscTimeSolverBase(MPI_Comm comm, Config const& cfg) {
    CHKERRTHROW(TSCreate(comm, &ts_));
    CHKERRTHROW(TSSetProblemType(ts_, TS_NONLINEAR));
    CHKERRTHROW(TSSetExactFinalTime(ts_, TS_EXACTFINALTIME_MATCHSTEP));
    CHKERRTHROW(TSSetFromOptions(ts_));

    auto const& cfgcp = cfg.ts_checkpoint_config;

    // we need the output prefixes for reading their status files
    const char* fault_probe_output_prefix = 0;
    if (cfg.fault_probe_output) {
        auto const& oc = *cfg.fault_probe_output;
        fault_probe_output_prefix = oc.prefix.c_str();
    }

    const char* domain_probe_output_prefix = 0;
    if (cfg.domain_probe_output) {
        auto const& oc = *cfg.domain_probe_output;
        domain_probe_output_prefix = oc.prefix.c_str();
    }

    const char* fault_output_prefix = 0;
    if (cfg.fault_output) {
        auto const& oc = *cfg.fault_output;
        fault_output_prefix = oc.prefix.c_str();
    }
    const char* domain_output_prefix = 0;
    if (cfg.domain_output) {
        auto const& oc = *cfg.domain_output;
        domain_output_prefix = oc.prefix.c_str();
    }

    CHKERRTHROW(ts_checkpoint_configure(
        ts_, cfgcp.save_directory.c_str(), cfgcp.frequency_step, cfgcp.frequency_cputime_minutes,
        cfgcp.frequency_time_physical, static_cast<int>(cfgcp.storage_type),
        cfgcp.storage_limited_size, fault_probe_output_prefix, domain_probe_output_prefix,
        fault_output_prefix, domain_output_prefix));

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

    // Set custom monitor
    // disable if -ts_monitor is set (then the default monitor will be used)
    // or if -disable_ts_custom_monitor is set
    PetscBool disableCustomTsMonitor = PETSC_FALSE;
    PetscBool defaultTsMonitorEnabled = PETSC_FALSE;
    CHKERRTHROW(
        PetscOptionsHasName(NULL, NULL, "-disable_custom_ts_monitor", &disableCustomTsMonitor));
    CHKERRTHROW(PetscOptionsHasName(NULL, NULL, "-ts_monitor", &defaultTsMonitorEnabled));

    if ((!disableCustomTsMonitor) && (!defaultTsMonitorEnabled)) {
        CHKERRTHROW(TSMonitorSet(ts_, PetscTimeSolverBase::customized_ts_monitor, NULL, NULL));
    }
}

PetscTimeSolverBase::~PetscTimeSolverBase() { TSDestroy(&ts_); }

PetscErrorCode PetscTimeSolverBase::customized_ts_monitor(TS ts, PetscInt step, PetscReal time,
                                                          Vec u, void* ctx) {
    PetscReal dt;
    CHKERRTHROW(TSGetTimeStep(ts, &dt));

    // Convert time and dt to the formatted string
    std::string formatted_time = tndm::format_time(time);
    std::string formatted_dt = tndm::format_time(dt);
    std::string current_datetime = tndm::get_current_date_time_string();

    // Print the step, formatted time, time step, and current date-time
    CHKERRTHROW(PetscPrintf(PETSC_COMM_WORLD, "%s Step %" PetscInt_FMT ": t = %s, dt = %s\n",
                            current_datetime.c_str(), step, formatted_time.c_str(),
                            formatted_dt.c_str()));
    return 0;
}

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
