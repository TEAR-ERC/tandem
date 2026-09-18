#!/bin/bash
# Generate test output for 3D Static Regression test
# Generates spherical_hole baseline output with static solver
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/3D"
MESH_DIR="${SCRIPT_DIR}/../../test_data/reference_meshes/3D"

cd "$CONFIG_DIR"

# sym link to the mesh file
ln -sf "$MESH_DIR/spherical_hole.msh" .

${EXECUTABLE_DIR}/app/static spherical_hole.toml \
--matrix_free yes --mg_strategy twolevel \
--mg_coarse_level 1 --output ${TEMP_TEST_RESULTS}/output3D \
--petsc -options_file mg_cheby.cfg > ${TEMP_TEST_RESULTS}/static_regression_3D.log

# clean up the sym link
rm spherical_hole.msh
