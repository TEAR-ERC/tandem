#!/bin/bash
# Generate test output for 2D volume tagging correctness test
# Generates outputs on a 2D mesh with different tags with static solver
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/2D"

cd "$CONFIG_DIR"


gmsh -2 volume_tagging_correctness.geo
${EXECUTABLE_DIR}/app/static volume_tagging_correctness.toml > ${TEMP_TEST_RESULTS}/volume_tagging_correctness_2D.log
rm volume_tagging_correctness.msh
