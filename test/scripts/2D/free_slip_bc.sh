#!/bin/bash
# Generate test output for checking correctness of free slip BC implementation
# Runs the free_slip block in extension and compression with static solver
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../../examples/elasticity/2d"
WORK_DIR="${TEMP_TEST_RESULTS}/free_slip_bc"

# The configs live in examples/; run on a copy so nothing is written there.
mkdir -p "$WORK_DIR"
cp "$CONFIG_DIR"/free_slip.{geo,lua,toml} "$WORK_DIR"
cd "$WORK_DIR"

gmsh -2 free_slip.geo
for scenario in free_slip_extension free_slip_compression; do
    ${EXECUTABLE_DIR}/app/static free_slip.toml --scenario ${scenario} \
        > ${TEMP_TEST_RESULTS}/${scenario}_2D.log
done
rm free_slip.msh
