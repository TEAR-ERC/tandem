#!/bin/bash
# Generate test output for the QD solve count test
# Runs bp1_ref (QD, not viscoelastic) with classic RK4, a scheme without FSAL, at a fixed
# step and with no output, and keeps the log with the solve and reassembly counts
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/2D"
MESH_DIR="${SCRIPT_DIR}/../../test_data/reference_meshes/2D"

cd "$CONFIG_DIR"

# sym link to the mesh file
ln -sf "$MESH_DIR/bp1_ref.msh" .

${EXECUTABLE_DIR}/app/tandem bp1_ref_QD_no_output.toml \
--petsc -options_file rk4_fixed.cfg > ${TEMP_TEST_RESULTS}/qd_solve_count.log

# clean up the sym link
rm bp1_ref.msh
