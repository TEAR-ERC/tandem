#!/bin/bash
# Generate test output for checking prescribed traction and traction-free BCs in 2D
# Runs the traction_patch block in tension and compression with static solver,
# once with ref_normal = [1, 0] and once with ref_normal = [-1, 0]
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../../examples/elasticity/2d"
WORK_DIR="${TEMP_TEST_RESULTS}/traction_patch_bc"

# The configs live in examples/; run on a copy so nothing is written there.
mkdir -p "$WORK_DIR"
cp "$CONFIG_DIR"/traction_patch.{geo,lua,toml} "$WORK_DIR"
cd "$WORK_DIR"

sed "s/^ref_normal = .*/ref_normal = [-1, 0]/" traction_patch.toml > traction_patch_flipped_normal.toml
grep -q "^ref_normal = \[-1, 0\]$" traction_patch_flipped_normal.toml

gmsh -2 traction_patch.geo
for scenario in traction_patch_tension traction_patch_compression; do
    ${EXECUTABLE_DIR}/app/static traction_patch.toml --scenario ${scenario} \
        > ${TEMP_TEST_RESULTS}/${scenario}_2D.log
    ${EXECUTABLE_DIR}/app/static traction_patch_flipped_normal.toml --scenario ${scenario} \
        > ${TEMP_TEST_RESULTS}/${scenario}_flipped_normal_2D.log
done
rm traction_patch.msh traction_patch_flipped_normal.toml
