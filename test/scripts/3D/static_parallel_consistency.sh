#!/bin/bash
# Generate test output for 3D Parallel Consistency test
# Generates spherical_hole outputs with varying MPI process counts (1,2,4,8)
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/3D"

cd "$CONFIG_DIR"

for i in 1 2 4 8; do
    gmsh -3 spherical_hole.geo
    mpirun --oversubscribe -n $i ${EXECUTABLE_DIR}/app/static spherical_hole.toml \
    --matrix_free yes --mg_strategy twolevel \
    --mg_coarse_level 1 --output ${TEMP_TEST_RESULTS}/parallel_output3D_$i \
    --petsc -options_file mg_cheby.cfg > ${TEMP_TEST_RESULTS}/static_parallel_consistency_3D.log
    rm spherical_hole.msh
done
