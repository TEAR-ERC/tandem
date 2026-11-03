#ifndef TS_UTIL_20211202_H
#define TS_UTIL_20211202_H

#include <petscts.h>

PetscErrorCode ts_checkpoint_configure(TS ts, const char* tsCheckpointSaveDirectory,
                                       int tsCheckpointFrequencyStep,
                                       double tsCheckpointFrequencyCputimeMinutes,
                                       double tsCheckpointFrequencyPhysical, int storageType,
                                       int tsCheckpointStorageLimitedSize);

/**
 * Callback invoked on rank 0 each time a checkpoint is written. It may print additional
 * lines (not starting with '#') into last_checkpoint.txt, directly after the checkpoint path.
 * Must be registered after ts_checkpoint_configure(); no-op if checkpointing is disabled.
 */
typedef PetscErrorCode (*TSCheckPointInfoFn)(PetscViewer viewer, void* ctx);
PetscErrorCode ts_checkpoint_set_info_callback(TS ts, TSCheckPointInfoFn fn, void* ctx);
PetscErrorCode ts_checkpoint(TS ts);
PetscErrorCode ts_checkpoint_restart(TS ts, const char* tsCheckpointLoadDirectory);

#endif // TS_UTIL_20211202_H
