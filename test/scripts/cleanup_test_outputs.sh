#!/bin/bash
# Cleanup temporary test outputs
set -e

SOURCE_DIR=$1

if [[ -z "$SOURCE_DIR" ]]; then
    echo "Usage: $0 <SOURCE_DIR>"
    echo ""
    echo "Arguments:"
    echo "  SOURCE_DIR - Path to source directory containing test data"
    exit 1
fi

TEMP_TEST_RESULTS=${SOURCE_DIR}/test/temp_test_results

if [[ -d "$TEMP_TEST_RESULTS" ]]; then
    rm -rf "$TEMP_TEST_RESULTS"
    echo "Cleaned up temporary test outputs at $TEMP_TEST_RESULTS"
else
    echo "No temporary test outputs found at $TEMP_TEST_RESULTS"
fi
