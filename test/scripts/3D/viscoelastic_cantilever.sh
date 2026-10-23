#!/bin/bash
# Generate test output for the viscoelastic cantilever rod creep-recovery test
# Runs examples/tandem/viscoelasticity/3d/cantilever_rod with tandem (QD), meshed
# with symmetry planes instead of the clamp (clamped = 0, see the .geo)
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../../examples/tandem/viscoelasticity/3d"
OPTIONS_DIR="${SCRIPT_DIR}/../../../examples/options"
WORK_DIR="${TEMP_TEST_RESULTS}/viscoelastic_cantilever"

# The configs live in examples/; run on a copy so nothing is written there.
# The probe files stay in WORK_DIR, where the test reads them.
mkdir -p "$WORK_DIR"
cp "$CONFIG_DIR"/cantilever_rod.{geo,lua,toml} "$WORK_DIR"
cd "$WORK_DIR"

gmsh -3 cantilever_rod.geo -setnumber clamped 0
${EXECUTABLE_DIR}/app/tandem cantilever_rod.toml \
    --petsc -options_file ${OPTIONS_DIR}/lu_mumps.cfg \
    > ${TEMP_TEST_RESULTS}/viscoelastic_cantilever.log
rm cantilever_rod.msh
