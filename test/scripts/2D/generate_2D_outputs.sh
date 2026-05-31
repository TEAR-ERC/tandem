#!/bin/bash
# Generate all 2D test outputs
# Manages: static_regression, parallel_consistency, convergence, seas_regression
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
POLYNOMIAL_DEGREE=$3
ENABLE_HDF5=$4

if [[ -z "$EXECUTABLE_DIR" || -z "$TEMP_TEST_RESULTS" || -z "$POLYNOMIAL_DEGREE" ]]; then
    echo "Usage: $0 <EXECUTABLE_DIR> <TEMP_TEST_RESULTS> <POLYNOMIAL_DEGREE>"
    echo ""
    echo "Arguments:"
    echo "  EXECUTABLE_DIR    -  Path to tandem/static executable"
    echo "  TEMP_TEST_RESULTS - Path to temporary test output directory"
    echo "  POLYNOMIAL_DEGREE - Polynomial degree used by the test build"
    exit 1
fi

if [[ ! "$POLYNOMIAL_DEGREE" =~ ^[0-9]+$ || "$POLYNOMIAL_DEGREE" -le 0 ]]; then
    echo "Error: POLYNOMIAL_DEGREE must be a positive integer. Got: '$POLYNOMIAL_DEGREE'"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Generating 2D reference outputs for p${POLYNOMIAL_DEGREE}..."

echo "Static regression (circular_hole)..."
bash "$SCRIPT_DIR/static_regression.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "Parallel consistency (circular_hole, 1/2/4/8 processes)..."
bash "$SCRIPT_DIR/static_parallel_consistency.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "Convergence study (circular_hole, varying h)..."
bash "$SCRIPT_DIR/convergence.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "SEAS regression (bp1_ref with tandem solver)..."
bash "$SCRIPT_DIR/seas_regression.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS" "$POLYNOMIAL_DEGREE"

if [[ "$ENABLE_HDF5" == "ON" ]]; then
    echo "HDF5 Probe writers (bp1_ref_QDGreen_probe_comparison with tandem solver)..."
    bash "$SCRIPT_DIR/HDF5_probe_writer.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"
else
    echo "Skipping data generation for HDF5 Probe writers (HDF5 not enabled in this build)."
fi
echo "Parallel volume tagging consistency test..."
bash "$SCRIPT_DIR/volume_tagging_parallel_consistency.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "Volume tagging correctness test..."
bash "$SCRIPT_DIR/volume_tagging_correctness.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS"

echo "2D reference outputs generated successfully."
