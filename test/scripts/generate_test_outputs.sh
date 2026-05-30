#!/bin/bash
# Generate all test outputs
# Manages data generation for 2D or 3D based on dimension argument
set -e

EXECUTABLE_DIR=$1
TEMP_TEST_RESULTS=$2
DOMAIN_DIMENSION=$3
POLYNOMIAL_DEGREE=$4
ENABLE_HDF5=$5

# Check if any required arguments are missing
if [[ -z "$EXECUTABLE_DIR" || -z "$TEMP_TEST_RESULTS" || -z "$DOMAIN_DIMENSION" ]]; then
    echo "Usage: $0 <EXECUTABLE_DIR> <TEMP_TEST_RESULTS> <DOMAIN_DIMENSION>"
    echo ""
    echo "Arguments:"
    echo "  EXECUTABLE_DIR    - Path to tandem/static executable"
    echo "  TEMP_TEST_RESULTS - Path to temporary test output directory"
    echo "  DOMAIN_DIMENSION  - 2 or 3"
    echo "  POLYNOMIAL_DEGREE - Polynomial degree used by the test build"
    exit 1
fi

# Validate the DOMAIN_DIMENSION argument
if [[ "$DOMAIN_DIMENSION" != "2" && "$DOMAIN_DIMENSION" != "3" ]]; then
    echo "Error: DOMAIN_DIMENSION must be '2' or '3'. Got: '$DOMAIN_DIMENSION'"
    exit 1
fi

if [[ ! "$POLYNOMIAL_DEGREE" =~ ^[0-9]+$ || "$POLYNOMIAL_DEGREE" -le 0 ]]; then
    echo "Error: POLYNOMIAL_DEGREE must be a positive integer. Got: '$POLYNOMIAL_DEGREE'"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$TEMP_TEST_RESULTS"

echo "Generating reference outputs for dimension: $DOMAIN_DIMENSION"
echo ""


echo "${DOMAIN_DIMENSION}D Reference Outputs"
bash "$SCRIPT_DIR/${DOMAIN_DIMENSION}D/generate_${DOMAIN_DIMENSION}D_outputs.sh" "$EXECUTABLE_DIR" "$TEMP_TEST_RESULTS" "$POLYNOMIAL_DEGREE" "$ENABLE_HDF5"
echo ""

echo "Reference outputs generated successfully!"
