#!/bin/bash
# Generate test output for 2D SEAS Regression test
# Generates bp1_ref outputs with tandem (SEAS) solver
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/2D"
MESH_DIR="${SCRIPT_DIR}/../../test_data/reference_meshes/2D"

cd "$CONFIG_DIR"

# sym link to the mesh file
ln -sf "$MESH_DIR/bp1_ref.msh" .


${EXECUTABLE_DIR}/app/tandem bp1_ref_QD.toml \
--petsc -options_file solver.cfg >> ${TEMP_TEST_RESULTS}/seas_regression_QD.log
${EXECUTABLE_DIR}/app/tandem bp1_ref_QDGreen.toml \
--petsc -options_file solver.cfg>> ${TEMP_TEST_RESULTS}/seas_regression_QDGreen.log

# clean up the sym link
rm bp1_ref.msh
