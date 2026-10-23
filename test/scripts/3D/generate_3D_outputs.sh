#!/bin/bash
# Generate all 3D test outputs
# Manages: static_regression, parallel_consistency, convergence, traction_bc,
# free_slip_bc
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2

if [[ -z "$EXECUTABLE_DIR" || -z "$TEMP_TEST_RESULTS" ]]; then
    echo "Usage: $0 <EXECUTABLE_DIR> <TEMP_TEST_RESULTS>"
    echo ""
    echo "Arguments:"
    echo "  EXECUTABLE_DIR    -  Path to tandem/static executable"
    echo "  TEMP_TEST_RESULTS - Path to temporary test output directory"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Generating 3D reference outputs..."

echo "Static regression (spherical_hole)..."
bash "$SCRIPT_DIR/static_regression.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "Parallel consistency (spherical_hole, 1/2/4/8 processes)..."
bash "$SCRIPT_DIR/static_parallel_consistency.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "Convergence study (spherical_hole, varying h)..."
bash "$SCRIPT_DIR/convergence.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "Traction BC correctness (cantilever rod)..."
bash "$SCRIPT_DIR/traction_bc.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "Free slip BC test (free_slip, extension and compression)..."
bash "$SCRIPT_DIR/free_slip_bc.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "3D reference outputs generated successfully."
