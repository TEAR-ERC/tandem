#!/bin/bash
# Generate test output for 2D Convergence test
# Generates circular_hole outputs with varying mesh sizes
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/2D"

cd "$CONFIG_DIR"

for h in 0.015625 0.03125 0.0625 0.125; do
    gmsh -2 circular_hole.geo -setnumber h $h -order 8 -o circular_hole.msh
    ${EXECUTABLE_DIR}/app/static circular_hole.toml > ${TEMP_TEST_RESULTS}/convergence_2D.log
    rm circular_hole.msh
done
