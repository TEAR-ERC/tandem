#!/bin/bash
# Generate test output for HDF5 probe writer test
# Generates bp1_ref outputs with tandem (SEAS) solver
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/2D"

cd "$CONFIG_DIR"

gmsh -2 bp1_ref.geo

${EXECUTABLE_DIR}/app/tandem bp1_ref_QDGreen_probe_comparison.toml \
--petsc -options_file solver.cfg >> ${TEMP_TEST_RESULTS}/bp1_ref_QDGreen_probe_comparison.log

rm bp1_ref.msh
