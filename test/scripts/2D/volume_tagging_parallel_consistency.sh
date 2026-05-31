#!/bin/bash
# Generate test output for 2D volume tagging parallel consistency test
# Generates outputs on a 2D mesh with different tags with static solver
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/2D"

cd "$CONFIG_DIR"


gmsh -2 volume_tagging_parallel_consistency.geo
for i in 1 2 4 8; do
    mpirun --oversubscribe -n $i ${EXECUTABLE_DIR}/app/static volume_tagging_parallel_consistency.toml --output ${TEMP_TEST_RESULTS}/parallel_volume_output2D_$i
done
rm volume_tagging_parallel_consistency.msh