#!/bin/bash
# Generate test output for checking correctness of traction BC implementation
# Generates cantilever_rod baseline output with static solver
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../../reference_configs/3D"

cd "$CONFIG_DIR"

gmsh -3 cantilever_rod.geo
${EXECUTABLE_DIR}/app/static cantilever_rod.toml \
--output ${TEMP_TEST_RESULTS}/cantilever_rod \
--petsc -options_file mg_cheby.cfg

rm cantilever_rod.msh
