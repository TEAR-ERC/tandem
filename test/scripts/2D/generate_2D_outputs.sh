#!/bin/bash
# Generate all 2D test outputs
# Manages: static_regression, parallel_consistency, convergence, seas_regression
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

echo "Generating 2D reference outputs..."

echo "Static regression (circular_hole)..."
bash "$SCRIPT_DIR/static_regression.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "Parallel consistency (circular_hole, 1/2/4/8 processes)..."
bash "$SCRIPT_DIR/static_parallel_consistency.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "Convergence study (circular_hole, varying h)..."
bash "$SCRIPT_DIR/convergence.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "SEAS regression (bp1_ref with tandem solver)..."
bash "$SCRIPT_DIR/seas_regression.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "2D reference outputs generated successfully."
