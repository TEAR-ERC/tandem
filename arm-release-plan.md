# ARM Release Image — Implementation Plan

## Goal

Produce a multi-arch release image (`linux/amd64` + `linux/arm64`) so that Apple Silicon (M1/M2/M3/M4) users can pull `tandem:latest` and run parallel simulations natively — no Rosetta, no QEMU overhead.

---

## Background & Key Differences for ARM

| Concern | x86-64 (current) | aarch64 (new) |
|---|---|---|
| Platform flag | `linux/amd64` | `linux/arm64` |
| Compiler arch flag | `-march=core-avx2` (implicit via host) | `-march=neoverse-v2` (explicit) |
| BLAS | OpenBLAS (already used — good) | OpenBLAS (same, no MKL on ARM) |
| SIMD codegen | libxsmm detects AVX2 | libxsmm detects NEON/SVE — same source, auto-detects |
| Runtime BLAS pkg | `libopenblas0-pthread` | `libopenblas0-pthread` (same, available on ARM Ubuntu) |
| `libparmetis4.0` | apt package | apt package (available on arm64 Ubuntu 24.04) |
| GitHub runner | `ubuntu-latest` (x86) | `ubuntu-24.04-arm` (native ARM runner) |

The base image (`tandem-base`) compiles PETSc + libxsmm from source, so both will compile correctly for ARM as long as the platform and compiler flags are set right.

---

## Architecture Decision: Multi-arch Manifest

Use Docker buildx multi-platform builds to produce a **single image tag** (`tandem:latest`) that contains both `amd64` and `arm64` layers. Docker automatically pulls the correct one based on the host architecture. This is the standard OCI multi-arch pattern.

```
tandem:latest
  ├── linux/amd64  (existing)
  └── linux/arm64  (new)
```

Same approach for `tandem-base`.

---

## Implementation Steps

### Phase 1 — Extend `docker/base/Dockerfile` for ARM

**File: `docker/base/Dockerfile`**

Two changes:

1. **Compiler flags**: libxsmm's `make generator` auto-detects architecture — no change needed.
2. **PETSc configure**: add `-march=neoverse-v2` via `COPTFLAGS`/`CXXOPTFLAGS` only when building on aarch64. Use a shell conditional:

```dockerfile
ARG TARGETARCH
# ... existing content ...
RUN ARCH_FLAGS=$([ "$TARGETARCH" = "arm64" ] && echo "-march=neoverse-v2" || echo "-march=core-avx2") && \
    ./configure \
      ...
      COPTFLAGS="-g -O3 $ARCH_FLAGS" CXXOPTFLAGS="-g -O3 $ARCH_FLAGS" \
      ...
```

`TARGETARCH` is automatically set by Docker buildx when building multi-platform — no `ARG` declaration needed beyond accepting it.

**`build-dependencies.yml`**: add `linux/arm64` to platforms and push both:
```yaml
platforms: linux/amd64,linux/arm64
```

### Phase 2 — Extend `docker/release/Dockerfile.all` for ARM

**File: `docker/release/Dockerfile.all`**

1. The `FROM --platform=linux/amd64 ${BASE_IMAGE}` lines hard-code amd64. Remove the explicit `--platform` annotation so buildx picks the correct variant automatically:
   ```dockerfile
   FROM ${BASE_IMAGE} AS builder
   # ...
   FROM ubuntu:24.04
   ```

2. Add architecture-aware cmake flags. Tandem's cmake invocation currently has no `-march` flag. Add via `CXXFLAGS`/`CFLAGS`:
   ```dockerfile
   ARG TARGETARCH
   RUN ARCH_FLAGS=$([ "$TARGETARCH" = "arm64" ] && echo "-march=neoverse-v2" || echo "-march=core-avx2") && \
       for DIM in 2 3; do \
           for DEG in 1 2 3 4 5 6; do \
               cmake -G Ninja /opt/tandem/tandem \
                   -DCMAKE_CXX_COMPILER=mpicxx \
                   -DCMAKE_C_COMPILER=mpicc \
                   -DCMAKE_PREFIX_PATH="${PETSC_INSTALL_DIR}" \
                   -DCMAKE_CXX_FLAGS="$ARCH_FLAGS" \
                   -DCMAKE_C_FLAGS="$ARCH_FLAGS" \
                   -DDOMAIN_DIMENSION=${DIM} \
                   -DPOLYNOMIAL_DEGREE=${DEG} \
               && ninja ...; \
           done; \
       done
   ```

3. Runtime stage: `ubuntu:24.04` already works on arm64 — no change. All apt packages (`libopenblas0-pthread`, `libparmetis4.0`, etc.) are available on arm64 Ubuntu 24.04.

### Phase 3 — Local Build & Test (Mac M chip)

**New file: `docker/release/build_arm.sh`**

```bash
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building tandem image for linux/arm64..."
docker buildx build --platform linux/arm64 \
    -f ${SCRIPT_DIR}/Dockerfile.all \
    -t tandem:arm64-test \
    --load ${SCRIPT_DIR}
echo "==> Done: tandem:arm64-test"
```

**Smoke test on Mac:**
```bash
# Run a quick help check on each binary
docker run --rm --platform linux/arm64 tandem:arm64-test \
    /bin/bash -c 'for b in /opt/tandem/bin/tandem_*; do echo "==> $b"; $b --help 2>&1 | head -2; done'

# Run a 2D example with MPI
docker run --rm --platform linux/arm64 tandem:arm64-test \
    mpirun -n 2 tandem_2d_2p /work/examples/...
```

**Expected result**: all 24 binaries (`tandem_Xd_Yp`, `static_Xd_Yp`) run without `Illegal instruction` or missing library errors.

### Phase 4 — Unified multi-arch local build

Once arm64 works, also update `build_all.sh` to produce both platforms in one push:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
    -f ${SCRIPT_DIR}/Dockerfile.all \
    -t tandem:latest \
    --push ${SCRIPT_DIR}       # --push required for multi-arch (no --load)
```

Note: multi-arch manifests require `--push` to a registry; `--load` only works for a single platform. For local single-arch testing, `--load` with one platform is fine.

### Phase 5 — GitHub Actions

**File: `.github/workflows/release-images.yml`**

Three changes:

1. **Add QEMU for cross-platform emulation** (needed on x86 GitHub runners to build arm64, OR use native ARM runners):

   **Option A — Native ARM runner (recommended, faster):** Add a separate job that runs on `ubuntu-24.04-arm`:
   ```yaml
   build-tandem-arm:
     name: Build tandem image (arm64)
     runs-on: ubuntu-24.04-arm        # GitHub native ARM runner
     ...
     - name: Build and push tandem image (arm64)
       uses: docker/build-push-action@v6
       with:
         platforms: linux/arm64
         push: true
         tags: ${{ steps.meta.outputs.tags }}-arm64   # temp per-arch tag
   ```

   **Option B — QEMU on x86 runner (simpler workflow, ~3× slower build):**
   ```yaml
   - name: Set up QEMU
     uses: docker/setup-qemu-action@v3
   - name: Build and push tandem image
     with:
       platforms: linux/amd64,linux/arm64
   ```

2. **Create multi-arch manifest** (if using Option A with parallel jobs): use `docker buildx imagetools create` in a final `merge` job to combine the per-arch digests into a single manifest list.

3. **Cache**: split cache keys by platform to avoid amd64/arm64 layer collisions:
   ```yaml
   cache-from: type=gha,scope=tandem-${{ matrix.platform }}
   cache-to: type=gha,mode=max,scope=tandem-${{ matrix.platform }}
   ```

**Recommended workflow structure:**
```
jobs:
  build-tandem-amd64   (ubuntu-latest)        → ghcr.io/.../tandem:sha-amd64
  build-tandem-arm64   (ubuntu-24.04-arm)     → ghcr.io/.../tandem:sha-arm64
  merge-manifest        (ubuntu-latest, needs both above)
                        → ghcr.io/.../tandem:latest  (multi-arch manifest)
```

---

## File Change Summary

| File | Change |
|---|---|
| `docker/base/Dockerfile` | Add `TARGETARCH`-conditional `COPTFLAGS`/`CXXOPTFLAGS`; remove hardcoded amd64 platform |
| `docker/release/Dockerfile.all` | Remove `--platform=linux/amd64` from `FROM`; add `TARGETARCH`-conditional `CMAKE_CXX_FLAGS` |
| `docker/release/build_arm.sh` | New script: local arm64 single-arch build |
| `docker/release/build_all.sh` | Add multi-arch `--push` variant (separate target) |
| `.github/workflows/release-images.yml` | Add arm64 job (native runner), merge-manifest job |
| `.github/workflows/build-dependencies.yml` | Add `linux/arm64` to platforms for base image |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `libparmetis4.0` not available on arm64 | Verify with `docker run --platform linux/arm64 ubuntu:24.04 apt-cache show libparmetis4.0` before coding; fall back to building ParMETIS from source if needed |
| libxsmm JIT on ARM falls back to scalar | Acceptable — still correct. Check with `LIBXSMM_VERBOSE=1` at runtime |
| PETSc downloads fail for ARM (no binary) | PETSc always compiles from source in our Dockerfile — not an issue |
| `ubuntu-24.04-arm` runner not available in repo | Check repo/org settings; fall back to QEMU Option B |
| Multi-arch `--load` limitation for local dev | Use `--load --platform linux/arm64` for local testing, `--push` for multi-arch release |

---

## Success Criteria

- [ ] `docker run --rm --platform linux/arm64 tandem:latest tandem_2d_2p --help` exits 0 on an M-chip Mac with no Rosetta
- [ ] `mpirun -n 4 tandem_2d_2p <example_input>` completes a 2D simulation in the container
- [ ] `docker manifest inspect ghcr.io/tear-erc/tandem/tandem:latest` shows both `amd64` and `arm64` entries
- [ ] CI passes on both PR branch and a version tag push
