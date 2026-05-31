#!/bin/bash
# Generate test output for HDF5 probe writer test
# Generates bp1_ref outputs with tandem (SEAS) solver
# Runs two separate simulations: one for CSV output, one for HDF5 output
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/2D"
MESH_DIR="${SCRIPT_DIR}/../../test_data/reference_meshes/2D"

cd "$CONFIG_DIR"

# sym link to the mesh file
ln -sf "$MESH_DIR/bp1_ref.msh" .

# Run CSV output simulation
${EXECUTABLE_DIR}/app/tandem bp1_ref_QDGreen_probe_comparison_csv.toml \
--petsc -options_file solver.cfg >> ${TEMP_TEST_RESULTS}/bp1_ref_QDGreen_probe_comparison_csv.log 2>&1

# Run HDF5 output simulation
${EXECUTABLE_DIR}/app/tandem bp1_ref_QDGreen_probe_comparison_hdf5.toml \
--petsc -options_file solver.cfg >> ${TEMP_TEST_RESULTS}/bp1_ref_QDGreen_probe_comparison_hdf5.log 2>&1

# clean up the sym link
rm bp1_ref.msh
