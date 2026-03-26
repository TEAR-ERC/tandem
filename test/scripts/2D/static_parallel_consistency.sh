#!/bin/bash
# Generate test output for 2D Parallel Consistency test
# Generates circular_hole outputs with varying MPI process counts (1,2,4,8)
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/2D"

cd "$CONFIG_DIR"

if [[ ! -f "circular_hole.msh" ]]; then
    gmsh -2 circular_hole.geo
fi

for i in 1 2 4 8; do
    mpirun --oversubscribe -n $i ${EXECUTABLE_DIR}/app/static circular_hole.toml \
    --matrix_free yes --mg_strategy twolevel \
    --mg_coarse_level 1 --output ${TEMP_TEST_RESULTS}/parallel_output2D_$i \
    --petsc -options_file mg_cheby.cfg
    rm circular_hole.msh
done

