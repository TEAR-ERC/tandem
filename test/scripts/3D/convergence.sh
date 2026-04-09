#!/bin/bash
# Generate test output for 3D Convergence test
# Generates spherical_hole outputs with varying mesh sizes
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/3D"

cd "$CONFIG_DIR"

for h in 0.125 0.25 0.5; do
    gmsh -3 spherical_hole.geo -setnumber h $h -order 8 -o spherical_hole.msh
    ${EXECUTABLE_DIR}/app/static spherical_hole.toml > ${TEMP_TEST_RESULTS}/convergence_3D.log
    rm spherical_hole.msh
done
