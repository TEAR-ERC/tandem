#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Build the base image once (PETSc + Eigen + Tandem source)
echo "==> Building base image..."
docker buildx build --platform linux/amd64 \
    -f ${SCRIPT_DIR}/Dockerfile.base \
    -t tandem-base:latest \
    --load ${SCRIPT_DIR}

# Step 2: Build all 12 individual images
for DIM in 2 3; do
    for DEG in 1 2 3 4 5 6; do
        echo "==> Building tandem:${DIM}d-${DEG}p ..."
        docker buildx build --platform linux/amd64 \
            --build-arg DIM=$DIM \
            --build-arg DEG=$DEG \
            -f ${SCRIPT_DIR}/Dockerfile.release \
            -t tandem:${DIM}d-${DEG}p \
            --load ${SCRIPT_DIR}
        echo "==> Done: tandem:${DIM}d-${DEG}p"
    done
done

# Step 3: Build the all-in-one image
echo "==> Building all-in-one image tandem:all ..."
docker buildx build --platform linux/amd64 \
    -f ${SCRIPT_DIR}/Dockerfile.all \
    -t tandem:all \
    --load ${SCRIPT_DIR}
echo "==> Done: tandem:all"

echo ""
echo "All images built:"
docker images | grep "^tandem"
