#!/bin/bash
# Generate test output for the viscoelastic simple shear test
# Runs examples/tandem/viscoelasticity/2d/simple_shear with tandem (QD), once per
# imposed strain history, and keeps the center probe of each run
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../../examples/tandem/viscoelasticity/2d"
OPTIONS_DIR="${SCRIPT_DIR}/../../../examples/options"
WORK_DIR="${TEMP_TEST_RESULTS}/viscoelastic_simple_shear"

# The configs live in examples/; run on a copy so nothing is written there.
mkdir -p "$WORK_DIR"
cp "$CONFIG_DIR"/simple_shear.{geo,lua,toml} "$WORK_DIR"
cd "$WORK_DIR"

gmsh -2 simple_shear.geo
# Direct solve: with a Krylov solver the error floor is the solver tolerance,
# not the time integration (see viscoelastic_simple_shear_config in conftest.py).
for case in constant_strain_without_drop constant_strain_with_drop \
            constant_strain_rate_without_drop constant_strain_rate_with_drop \
            sinusoidal_without_drop sinusoidal_with_drop; do
    ${EXECUTABLE_DIR}/app/tandem simple_shear.toml --scenario ${case} \
        --petsc -options_file ${OPTIONS_DIR}/lu_mumps.cfg \
        > ${TEMP_TEST_RESULTS}/viscoelastic_simple_shear_${case}.log
    mv blkst_center.csv ${TEMP_TEST_RESULTS}/viscoelastic_simple_shear_${case}_center.csv
done
rm simple_shear.msh
