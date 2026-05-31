#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the tandem image (all 12 configurations) using the shared base
echo "==> Building tandem image..."
docker buildx build --platform linux/amd64 \
    -f ${SCRIPT_DIR}/Dockerfile.all \
    -t tandem:latest \
    --load ${SCRIPT_DIR}
echo "==> Done: tandem:latest"

echo ""
docker images | grep "^tandem"
