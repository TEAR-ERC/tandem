#!/bin/bash
# Generate test output for the viscoelastic manufactured solution test
# Runs examples/tandem/viscoelasticity/2d/mms with tandem (QD):
#   spatial study:  scenario mms (trigonometric) at theta = 0.01, varying resolution
#   temporal study: scenario mms_poly (polynomial) at resolution 0.25, varying theta
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../../examples/tandem/viscoelasticity/2d"
OPTIONS_DIR="${SCRIPT_DIR}/../../../examples/options"
WORK_DIR="${TEMP_TEST_RESULTS}/viscoelastic_mms"

# The configs live in examples/; run on a copy so nothing is written there.
mkdir -p "$WORK_DIR"
cp "$CONFIG_DIR"/mms.{lua,toml} "$WORK_DIR"
cd "$WORK_DIR"

# Direct solve, so that the errors are discretisation errors only.
SOLVER="--petsc -options_file ${OPTIONS_DIR}/lu_mumps.cfg"

for r in 0.25 0.125 0.0625; do
    ${EXECUTABLE_DIR}/app/tandem mms.toml --scenario mms --resolution ${r} ${SOLVER} \
        > ${TEMP_TEST_RESULTS}/viscoelastic_mms_space_r${r}.log
done

# theta is a lua member: sweep it on a copy of mms.lua
for theta in 0.4 0.2 0.1 0.05; do
    sed "s/^MMS.theta = .*/MMS.theta = ${theta}/" mms.lua > mms_theta.lua
    grep -q "^MMS.theta = ${theta}$" mms_theta.lua
    ${EXECUTABLE_DIR}/app/tandem mms.toml --lib mms_theta.lua --scenario mms_poly \
        --resolution 0.25 ${SOLVER} \
        > ${TEMP_TEST_RESULTS}/viscoelastic_mms_time_theta${theta}.log
done
rm mms_theta.lua
