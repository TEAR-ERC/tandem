#!/bin/bash
# Generate test output for 2D Static Regression test
# Generates circular_hole baseline output with static solver
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/2D"
MESH_DIR="${SCRIPT_DIR}/../../test_data/reference_meshes/2D"

cd "$CONFIG_DIR"

# sym link to the mesh file
ln -sf "$MESH_DIR/circular_hole.msh" .

${EXECUTABLE_DIR}/app/static circular_hole.toml \
--matrix_free yes --mg_strategy twolevel \
--mg_coarse_level 1 --output ${TEMP_TEST_RESULTS}/output2D \
--petsc -options_file mg_cheby.cfg > ${TEMP_TEST_RESULTS}/static_regression_2D.log

# clean up the sym link
rm circular_hole.msh
