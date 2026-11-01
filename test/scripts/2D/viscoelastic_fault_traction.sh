#!/bin/bash
# Generate test output for the viscoelastic fault traction test
# Runs examples/tandem/viscoelasticity/2d/fault_shear with tandem (QD): the antiplane
# variant (PoissonViscoelasticity) with classic RK4 at a fixed step, a scheme without FSAL,
# and with Dormand-Prince 5(4), which has FSAL, and the inplane variant (Viscoelasticity)
# with classic RK4
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../../examples/tandem/viscoelasticity/2d"
OPTIONS_DIR="${SCRIPT_DIR}/../../../examples/options"
WORK_DIR="${TEMP_TEST_RESULTS}/viscoelastic_fault_traction"

# Fixed step theta * tau_relax = 0.05 * 2.36682e10 / 30
RK4="-ts_type rk -ts_rk_type 4 -ts_adapt_type none -ts_dt 3.9447e7"
RK5DP="-ts_type rk -ts_rk_type 5dp"

# The configs live in examples/; run each case on a copy so nothing is written there.
# The probe files stay in WORK_DIR/<case>, where the test reads them.
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
gmsh -2 "$CONFIG_DIR/fault_shear.geo" -o fault_shear.msh

run_case() {
    local name=$1 type=$2 scenario=$3 ts=$4
    mkdir -p "$name"
    cp "$CONFIG_DIR"/fault_shear.{lua,toml} fault_shear.msh "$name"
    (cd "$name" && ${EXECUTABLE_DIR}/app/tandem fault_shear.toml --type "$type" \
        --scenario "$scenario" --petsc -options_file ${OPTIONS_DIR}/lu_mumps.cfg $ts \
        > ${TEMP_TEST_RESULTS}/viscoelastic_fault_traction_${name}.log)
    rm "$name/fault_shear.msh"
}

run_case antiplane_rk4 poisson_viscoelasticity antiplane "$RK4"
run_case antiplane_5dp poisson_viscoelasticity antiplane "$RK5DP"
run_case inplane_rk4 viscoelasticity inplane "$RK4"
rm fault_shear.msh
