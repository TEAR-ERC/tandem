#!/bin/bash
# Build a linux/arm64 release image locally (for Apple Silicon / M-chip Macs).
# Runs natively on M1/M2/M3/M4 — no QEMU overhead.
# Note: -march=neoverse-v2 is not used because Docker Desktop on Apple Silicon
# does not expose SVE to the Linux VM. Compiled with default arm64 baseline.
#
# Usage (from repo root or any path):
#   ./docker/release/build_arm.sh
#
# After the build completes, test with:
#   docker run --rm tandem:arm64-test tandem_2d_2p --help
#   docker run --rm tandem:arm64-test mpirun -n 4 tandem_2d_2p <input_file>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PETSC_VERSION="3.22.5"
BASE_IMAGE_LOCAL="tandem-base:arm64-local"
RELEASE_IMAGE="tandem:arm64-test"

echo "==> Step 1/2: Building ARM base image (PETSc ${PETSC_VERSION})..."
echo "    This compiles PETSc from source — expect 20-40 min on first run."
docker buildx build \
    --platform linux/arm64 \
    --build-arg PETSC_VERSION="${PETSC_VERSION}" \
    --build-arg CC=gcc-13 \
    --build-arg CXX=g++-13 \
    -f "${REPO_ROOT}/docker/base/Dockerfile" \
    -t "${BASE_IMAGE_LOCAL}" \
    --load \
    "${REPO_ROOT}"

echo ""
echo "==> Step 2/2: Building ARM release image (all 12 configs)..."
echo "    This compiles 24 Tandem binaries — expect 15-30 min on first run."
docker buildx build \
    --platform linux/arm64 \
    --build-arg BASE_IMAGE="${BASE_IMAGE_LOCAL}" \
    -f "${SCRIPT_DIR}/Dockerfile.all" \
    -t "${RELEASE_IMAGE}" \
    --load \
    "${REPO_ROOT}"

echo ""
echo "==> Done: ${RELEASE_IMAGE}"
echo ""
echo "Quick smoke test:"
echo "  docker run --rm ${RELEASE_IMAGE} tandem_2d_2p --help"
echo ""
echo "MPI parallel test (replace <input> with an actual .lua file):"
echo "  docker run --rm ${RELEASE_IMAGE} mpirun -n 4 tandem_2d_2p <input>"
