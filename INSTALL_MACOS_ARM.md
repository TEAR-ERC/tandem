# Native Tandem Installation on Apple Silicon (M1/M2/M3/M4)

This guide installs Tandem and all its dependencies natively on macOS Apple Silicon.
Native installation gives full access to all CPU cores with no virtualisation overhead.

---

## Prerequisites

### 1. Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. System dependencies

```bash
brew install \
    gcc@13 \
    cmake ninja \
    open-mpi \
    metis parmetis \
    eigen \
    lua@5.3 \
    python3 \
    wget git
```

> **Note:** `libxsmm` 1.17 is x86-only and is **skipped** on ARM. Tandem's yateto kernel
> generator automatically falls back to BLAS — correctness is not affected.

---

## 1. Install PETSc from source

PETSc must be compiled from source. It downloads and builds MUMPS and ScaLAPACK automatically.

```bash
export PETSC_INSTALL_DIR="$HOME/opt/petsc"
mkdir -p "$PETSC_INSTALL_DIR"

wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.22.5.tar.gz
tar -xf petsc-3.22.5.tar.gz
cd petsc-3.22.5

./configure \
    --with-fortran-bindings=0 \
    --with-debugging=0 \
    --with-memalign=32 \
    --with-64-bit-indices \
    CC=mpicc CXX=mpicxx FC=mpif90 \
    COPTFLAGS="-g -O3" CXXOPTFLAGS="-g -O3" \
    --download-mumps \
    --download-scalapack \
    --prefix="$PETSC_INSTALL_DIR"

make PETSC_DIR=$(pwd) PETSC_ARCH=arch-linux-c-opt -j$(nproc)
make PETSC_DIR=$(pwd) PETSC_ARCH=arch-linux-c-opt install

cd ..
rm -rf petsc-3.22.5 petsc-3.22.5.tar.gz
```

> **Tip:** `nproc` on macOS may return 1 — use `sysctl -n hw.logicalcpu` to get the real
> core count. Replace `-j$(nproc)` with e.g. `-j11` for an M3.

Add PETSc to your shell environment (add to `~/.zshrc` or `~/.bashrc`):

```bash
export PETSC_INSTALL_DIR="$HOME/opt/petsc"
export LD_LIBRARY_PATH="$PETSC_INSTALL_DIR/lib:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="$PETSC_INSTALL_DIR/lib:$DYLD_LIBRARY_PATH"
```

---

## 2. Install Python dependencies

Tandem's yateto kernel generator requires numpy at build time:

```bash
pip3 install numpy
```

---

## 3. Clone Tandem

```bash
mkdir -p ~/opt/tandem
cd ~/opt/tandem
git clone https://github.com/TEAR-ERC/tandem.git
cd tandem
git checkout dmay/staging
git submodule update --init
```

---

## 4. Build Tandem

Choose your dimension (`2` or `3`) and polynomial degree (`1`–`6`).
The example below builds 2D degree 6 (`tandem_2d_6p`).

```bash
DIM=2
DEG=6

mkdir -p ~/opt/tandem/build_${DIM}d_${DEG}p
cd ~/opt/tandem/build_${DIM}d_${DEG}p

cmake -G Ninja ~/opt/tandem/tandem \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_PREFIX_PATH="$PETSC_INSTALL_DIR" \
    -DARCH=noarch \
    -DDOMAIN_DIMENSION=${DIM} \
    -DPOLYNOMIAL_DEGREE=${DEG}

ninja -j$(sysctl -n hw.logicalcpu)
```

> **`-DARCH=noarch`** is required on ARM. The default (`hsw`) injects `-march=haswell`
> which is x86-only and will fail to compile on Apple Silicon.

Binaries are placed at:
- `app/tandem` → the main Tandem solver
- `app/static` → the static pre-processor

Optionally copy them to a convenient location:

```bash
mkdir -p ~/opt/tandem/bin
cp app/tandem ~/opt/tandem/bin/tandem_${DIM}d_${DEG}p
cp app/static ~/opt/tandem/bin/static_${DIM}d_${DEG}p
```

Add to your PATH (in `~/.zshrc` or `~/.bashrc`):

```bash
export PATH="$HOME/opt/tandem/bin:$PATH"
```

---

## 5. Run a parallel simulation

OpenMPI on macOS uses all available cores by default. Run with:

```bash
mpirun -n 6 tandem_2d_6p your_input.toml
```

To verify all cores are in use, open Activity Monitor → CPU tab and look for multiple
`tandem_2d_6p` processes, or use:

```bash
# in a second terminal while the simulation runs
top -pid $(pgrep -d',' tandem_2d_6p)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `error: unknown value 'haswell' for '-march'` | You forgot `-DARCH=noarch` in cmake |
| `PETSc not found` | Check `PETSC_INSTALL_DIR` is exported and `pkg-config` can find `$PETSC_INSTALL_DIR/lib/pkgconfig` |
| `libparmetis not found` | Run `brew install parmetis` and add `-DCMAKE_PREFIX_PATH="$(brew --prefix parmetis)"` |
| `Lua not found` | Run `brew install lua@5.3` and add `-DCMAKE_PREFIX_PATH="$(brew --prefix lua@5.3)"` |
| MPI runs on 1 core only | Add `--map-by core --bind-to core` to `mpirun` |
| `nproc` returns 1 | Use `sysctl -n hw.logicalcpu` instead to get real core count |

---

## Build all 12 configurations

To build all dimension/degree combinations in one go:

```bash
export PETSC_INSTALL_DIR="$HOME/opt/petsc"
mkdir -p ~/opt/tandem/bin

for DIM in 2 3; do
    for DEG in 1 2 3 4 5 6; do
        echo "==> Building ${DIM}D degree ${DEG}..."
        mkdir -p ~/opt/tandem/build_${DIM}d_${DEG}p
        cd ~/opt/tandem/build_${DIM}d_${DEG}p
        cmake -G Ninja ~/opt/tandem/tandem \
            -DCMAKE_CXX_COMPILER=mpicxx \
            -DCMAKE_C_COMPILER=mpicc \
            -DCMAKE_PREFIX_PATH="$PETSC_INSTALL_DIR" \
            -DARCH=noarch \
            -DDOMAIN_DIMENSION=${DIM} \
            -DPOLYNOMIAL_DEGREE=${DEG}
        ninja -j$(sysctl -n hw.logicalcpu)
        cp app/tandem ~/opt/tandem/bin/tandem_${DIM}d_${DEG}p
        cp app/static ~/opt/tandem/bin/static_${DIM}d_${DEG}p
    done
done
```
