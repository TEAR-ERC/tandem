#!/bin/bash
# Generate test output for checking correctness of free slip BC implementation
# Runs the free_slip block in extension and compression with static solver
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/2D"

cd "$CONFIG_DIR"

gmsh -2 free_slip.geo
for scenario in free_slip_extension free_slip_compression; do
    ${EXECUTABLE_DIR}/app/static free_slip.toml --scenario ${scenario} \
        > ${TEMP_TEST_RESULTS}/${scenario}_2D.log
done
rm free_slip.msh
