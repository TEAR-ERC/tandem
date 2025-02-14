#ifndef TS_UTIL_20211202_H
#define TS_UTIL_20211202_H

#include <petscts.h>

PetscErrorCode
ts_checkpoint_configure(TS ts, const char* tsCheckpointSaveDirectory, int tsCheckpointFrequencyStep,
                        double tsCheckpointFrequencyCputimeMinutes,
                        double tsCheckpointFrequencyPhysical, int storageType,
                        int tsCheckpointStorageLimitedSize, const char* fault_probe_output_prefix,
                        const char* domain_probe_output_prefix, const char* fault_output_prefix,
                        const char* domain_output_prefix);
PetscErrorCode ts_checkpoint(TS ts);
PetscErrorCode ts_checkpoint_restart(TS ts, const char* tsCheckpointLoadDirectory);

#endif // TS_UTIL_20211202_H
