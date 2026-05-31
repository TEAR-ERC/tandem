#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Build the base image once (PETSc + Eigen + Tandem source)
echo "==> Building base image..."
docker buildx build --platform linux/amd64 \
    -f ${SCRIPT_DIR}/Dockerfile.base \
    -t tandem-base:latest \
    --load ${SCRIPT_DIR}

# Step 2: Build the tandem image (all 12 configurations)
echo "==> Building tandem image..."
docker buildx build --platform linux/amd64 \
    -f ${SCRIPT_DIR}/Dockerfile.all \
    -t tandem:latest \
    --load ${SCRIPT_DIR}
echo "==> Done: tandem:latest"

echo ""
echo "Images built:"
docker images | grep -E "^tandem"
