#!/bin/bash
# Generate test output for the viscoelastic two-layer creep test
# Runs examples/tandem/viscoelasticity/2d/two_layer with tandem (QD): once with the
# relaxation time derived from viscosity / mu1, and once with a relaxation_time that
# contradicts it, which must be rejected (that run is expected to fail)
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../../examples/tandem/viscoelasticity/2d"
OPTIONS_DIR="${SCRIPT_DIR}/../../../examples/options"
WORK_DIR="${TEMP_TEST_RESULTS}/viscoelastic_two_layer"

# The configs live in examples/; run on a copy so nothing is written there.
# The probe files stay in WORK_DIR, where the test reads them.
mkdir -p "$WORK_DIR"
cp "$CONFIG_DIR"/two_layer.{geo,lua,toml} "$WORK_DIR"
cd "$WORK_DIR"

gmsh -2 two_layer.geo
${EXECUTABLE_DIR}/app/tandem two_layer.toml \
    --petsc -options_file ${OPTIONS_DIR}/lu_mumps.cfg \
    > ${TEMP_TEST_RESULTS}/viscoelastic_two_layer.log
${EXECUTABLE_DIR}/app/tandem two_layer.toml --scenario two_layer_inconsistent \
    --petsc -options_file ${OPTIONS_DIR}/lu_mumps.cfg \
    > ${TEMP_TEST_RESULTS}/viscoelastic_two_layer_inconsistent.log 2>&1 || true
rm two_layer.msh
