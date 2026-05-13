# STRUMPACK Transition Plan: Randomized Peeling via Matrix-Free H-Matrix Construction

## 1. Current State

### 1.1 What Is Done (HTool / MATHTOOL)

The H-matrix implementation using HTool (via PETSc's `MatHtool`) is **complete and compilable**:

* `HMatrixGreenFunction` (`app/form/HMatrixGreenFunction.{h,cpp}`) owns `D*(D-1)` HTool sub-matrices extracted from the dense GF, plus all permutation and scatter infrastructure.
* `SeasQDDiscreteGreenOperator` routes `update_traction()` through `H_.apply()` when `use_hmatrix = true`.
* `gf-hmatrix-validate` validates accuracy against the dense GF and reports compression ratio.
* The GF checkpoint must be regenerated under the current 32-bit PETSc build before the first test run (old 64-bit checkpoint is incompatible).

### 1.2 Goal of This Transition

Transition the off-diagonal compression backend from HTool to STRUMPACK in order to enable **matrix-free construction**: build the H-matrix using only black-box matrix-vector products (MatVecs) rather than entry-wise kernel evaluation. This matters because:

* The forward MatVec $Gv$ is already provided by the PDE solver.
* Entry-wise kernel access requires the full dense $G$ to already be assembled — eliminating the $O(N^2)$ benefit.
* Matrix-free construction bounds the construction cost by $O(k \cdot N \cdot T_{matvec})$, where $k$ is the number of random vectors used.

The **immediate deliverable** is `gf-strumpack-bench`: a benchmark binary that uses the dense $G$ as a stand-in for both $Gv$ and $G^Tw$, counting exactly how many MatVecs STRUMPACK needs to reach target accuracy. This number is the projected cost of the eventual adjoint-free construction path.

The **final goal** is to wire `StrumpackGFOperator` into `SeasQDDiscreteGreenOperator` as an optional backend (parallel to the existing HTool path), targeting **Np ~ 10⁶** degrees of freedom.

---

## 2. Environment Reality Check

### 2.1 What Is Installed

STRUMPACK **7.2.0** is available at `/home/ymagen/petsc/build`:
* Library: `lib/libstrumpack.so`
* CMake config: `lib/cmake/STRUMPACK/strumpack-config.cmake`
* C++ headers: `include/structured/StructuredMatrix.hpp`, `include/HODLR/HODLROptions.hpp`, etc.

### 2.2 Critical Gap: ButterflyPACK Is Not Enabled

The installed STRUMPACK was built with `-DSTRUMPACK_USE_BPACK=OFF`. This means:

* `dC_BPACK_wrapper.h` **does not exist** in this build.
* `d_c_bpack_construct_matvec_compute` and the entire ButterflyPACK Fortran-C interface are **unavailable**.
* The original plan's implementation path is blocked without rebuilding PETSc.

### 2.3 Sub-Matrix Structure: What Is Actually Compressed

Following the same decomposition as the existing HTool implementation, STRUMPACK will compress **not** the full 2800×1400 GF, but the `D*(D-1)` component sub-matrices `G_αβ` produced by `extract_component_submatrix`. Each `G_αβ` maps slip component β to traction component α across all `Np = N_el*nbf` nodes, and has shape `Np × Np` — **square**:

* **2D** (D=2, slip_D=1): 2 sub-matrices `G_00`, `G_10`, each **1400×1400** for BP1.
* **3D** (D=3, slip_D=2): 6 sub-matrices `G_00`, `G_01`, `G_10`, `G_11`, `G_20`, `G_21`, each **Np×Np**.

**Production target: Np ~ 10⁶.** All design decisions (MPI bridge, cluster tree construction) must scale to this regime. BP1 (Np=1400) is only the first validation case.

`StrumpackGFOperator` therefore owns a vector `S_ab_` of `D*(D-1)` STRUMPACK objects, mirroring `HMatrixGreenFunction::H_ab_`. The `planar_fault` optimisation (skip zero normal-traction components `G_0β`) carries over unchanged.

### 2.4 What Is Available: Native STRUMPACK StructuredMatrix API

STRUMPACK's own C++ `structured::StructuredMatrix<double>` interface is fully available and supports matrix-free construction. Because each `G_αβ` is **square**, all MF-capable formats apply:

| Format    | seq | MPI | Dense | Elem | MF | PMF | mult |
|-----------|-----|-----|-------|------|----|-----|------|
| BLR       | X   | X   | X     | X    |    |     | X    |
| HSS       | X   | X   | X     |      |    | X   | X    |
| HODLR     |     | X   | X     | X    | X  |     | X    |
| HODBF     |     | X   | X     | X    | X  |     | X    |
| BUTTERFLY |     | X   | X     | X    | X  |     | X    |

* **MF** = `construct_matrix_free()` — MatVec callback only (no element access). Directly measures the MatVec budget.
* **PMF** = `construct_partially_matrix_free()` — needs MatVec + element extraction. HSS is usable this way (we have the dense `G_αβ`), but PMF mixes entry access into the cost, so it is less useful for the pure MatVec count benchmark.
* **BLR** does not support MF at all — ruled out.
* **HODLR / HODBF / BUTTERFLY** all support pure MF and run in MPI. These are the primary targets.

**Recommended format for benchmarking:** `BUTTERFLY` (or `HODBF`). Both use randomized matrix-free construction and count adjoint+forward MatVecs explicitly. `HODLR` is also valid but constructs a simpler off-diagonal low-rank structure.

---

## 3. MPI Callback Bridge Design (Critical for Np ~ 10⁶)

### 3.1 Why the Bridge Approach Matters at Scale

STRUMPACK's MPI `construct_matrix_free` uses the `mult_1d_t` callback signature:

```cpp
void(Trans op,
     const DenseMatrix<double>& R,   // input, distributed by cdist (Trans::N) or rdist (Trans::T)
     DenseMatrix<double>& S,          // output, distributed by rdist (Trans::N) or cdist (Trans::T)
     const std::vector<int>& rdist,   // row distribution: rank p owns rows [rdist[p], rdist[p+1])
     const std::vector<int>& cdist)   // col distribution: rank p owns cols [cdist[p], cdist[p+1])
```

The callback is invoked collectively. Rank p receives in `R` the local slice of `nvec` random vectors corresponding to its column partition.

**Why `MPI_Allgatherv` per column is unacceptable at scale:**

Each callback call moves `nvec × Np × sizeof(double)` bytes across all ranks. At Np=10⁶, nvec=50:
```
50 × 10⁶ × 8 bytes = 400 MB per callback call
```
BUTTERFLY construction issues O(log(Np/leaf) × k) callback calls. The total extra communication grows as O(N²) — equivalent to assembling the full dense matrix repeatedly. This completely defeats the matrix-free construction goal.

**The zero-copy approach is required for production correctness.**

### 3.2 Matching the Cluster Tree to PETSc's Layout

The key insight: if we build the STRUMPACK cluster tree so that its 1D distribution **exactly matches PETSc's PETSC_DECIDE layout**, then `rdist[p] == perm_rstart_p` and `cdist[p] == perm_cstart_p` in every callback call. The local slice of `R` handed to rank p is then precisely the data PETSc expects for that rank's columns of `G_perm_ab` — **no redistribution needed**.

PETSc's PETSC_DECIDE assigns rank `p` rows:
```
rstart_p = p*(Np/P) + min(p, Np%P)
rend_p   = rstart_p + Np/P + (p < Np%P ? 1 : 0)
```

We build a `ClusterTree` encoding this partition as a binary tree over rank ranges `[lo, hi)`:

```cpp
// dist[r] = rstart_r (r=0..P), dist[P] = Np
ClusterTree build_petsc_tree(const std::vector<int>& dist, int lo, int hi) {
    ClusterTree t(dist[hi] - dist[lo]);
    if (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        t.c.push_back(build_petsc_tree(dist, lo, mid));
        t.c.push_back(build_petsc_tree(dist, mid, hi));
    }
    return t;
}
// Usage:
std::vector<int> dist(n_ranks + 1, 0);
for (int r = 0; r < n_ranks; ++r)
    dist[r+1] = dist[r] + Np/n_ranks + (r < Np%n_ranks ? 1 : 0);
// Verify against actual PETSc ownership
assert(dist[my_rank] == perm_rstart && dist[my_rank+1] == perm_rend);
ClusterTree row_tree = build_petsc_tree(dist, 0, n_ranks);
// For square G_αβ: col_tree == row_tree (same distribution)
```

### 3.3 Zero-Copy Callback Implementation

With the matching tree, the callback uses `VecPlaceArray`/`VecResetArray` to swap data pointers into pre-allocated PETSc Vecs — no allocation, no copy, no extra communication:

```cpp
// Pre-allocated once in build_one_s_matrix:
Vec x_work;  // local_n = perm_cend - perm_cstart, global Np
Vec y_work;  // local_m = perm_rend - perm_rstart, global Np
VecCreateMPIWithArray(comm_, 1, local_n, Np, nullptr, &x_work);
VecCreateMPIWithArray(comm_, 1, local_m, Np, nullptr, &y_work);

auto Amult = [&](Trans t,
                 const DenseMatrix<double>& R,
                 DenseMatrix<double>& S,
                 const std::vector<int>& rdist,
                 const std::vector<int>& cdist) {

    int nvec = R.cols();
    if (t == Trans::N) matvec_count_  += nvec;   // count vectors, not callbacks
    else               adjoint_count_ += nvec;

    for (int k = 0; k < nvec; ++k) {
        if (t == Trans::N) {
            // Trans::N: S = A * R
            // R distributed by cdist → cdist[p]==perm_cstart (guaranteed by tree)
            // S distributed by rdist → rdist[p]==perm_rstart (guaranteed by tree)
            VecPlaceArray(x_work, R.ptr(0, k));    // zero-copy: R.ptr(0,k) has local_n doubles
            VecPlaceArray(y_work, S.ptr(0, k));
            MatMult(G_perm_ab, x_work, y_work);    // PETSc handles cross-rank comm internally
            VecResetArray(x_work);
            VecResetArray(y_work);
        } else {
            // Trans::T: S = A^T * R
            // R distributed by rdist, S distributed by cdist
            VecPlaceArray(y_work, R.ptr(0, k));
            VecPlaceArray(x_work, S.ptr(0, k));
            MatMultTranspose(G_perm_ab, y_work, x_work);
            VecResetArray(x_work);
            VecResetArray(y_work);
        }
    }
};
```

`VecPlaceArray` replaces the Vec's data pointer without touching its parallel layout (ownership ranges, ghost comm). `MatMult` on the distributed `G_perm_ab` communicates only what the distributed multiply requires — which is unavoidable regardless of implementation.

### 3.4 MatVec Counter Semantics

We count **total random vectors** (`+= nvec` per callback invocation), not callback invocations. This is the right metric because in the real scenario each sample vector corresponds to one full PDE solve. The "Projected PDE-solver evaluations" in the benchmark output is:

```
Σ nvec  over all Trans::N callback invocations
```

---

## 4. Implementation Plan

### Phase 1: Build System

Add STRUMPACK to the benchmark target in `app/CMakeLists.txt`:

```cmake
find_package(STRUMPACK REQUIRED
    HINTS /home/ymagen/petsc/build/lib/cmake/STRUMPACK)

target_link_libraries(gf-strumpack-bench PRIVATE STRUMPACK::strumpack)
```

No `FindSTRUMPACK.cmake` module is needed — STRUMPACK ships its own CMake config.

### Phase 2: `StrumpackGFOperator` Class

Create `app/form/StrumpackGFOperator.{h,cpp}` following `HMatrixGreenFunction` closely. The key structural parallel: where `HMatrixGreenFunction` owns `H_ab_` (a vector of HTool `Mat`s), `StrumpackGFOperator` owns `S_ab_` (a vector of STRUMPACK `StructuredMatrix` objects), one per `(α,β)` pair.

**Header sketch:**

```cpp
#include <strumpack/structured/StructuredMatrix.hpp>
#include <strumpack/structured/StructuredOptions.hpp>
#include <petscmat.h>
#include <vector>
#include <memory>
#include <atomic>

namespace tndm {

class StrumpackGFOperator {
public:
    StrumpackGFOperator(Mat G_dense,
                        const std::vector<PetscReal>& local_coords,
                        PetscInt nbf, int D,
                        Vec s_proto, Vec t_proto,
                        MPI_Comm comm,
                        HMatrixConfig const& config);

    void apply(Vec s, Vec t) const;   // t += Σ_{α,β} S_αβ * s_β

    std::size_t total_matvec_count()  const { return matvec_count_; }
    std::size_t total_adjoint_count() const { return adjoint_count_; }

private:
    int      D_, slip_D_;
    PetscInt nbf_, N_el_;
    MPI_Comm comm_;
    HMatrixConfig config_;

    std::vector<std::unique_ptr<strumpack::structured::StructuredMatrix<double>>> S_ab_;

    std::vector<PetscInt> perm_;
    std::vector<Vec>        slip_spatial_, traction_spatial_;
    std::vector<VecScatter> scatter_s_, scatter_t_;

    mutable std::atomic<std::size_t> matvec_count_{0};
    mutable std::atomic<std::size_t> adjoint_count_{0};

    void build_spatial_permutation(const std::vector<PetscReal>& global_coords);
    // build_petsc_tree, extract_component_submatrix, build_spatially_permuted_submatrix:
    // identical to HMatrixGreenFunction counterparts
    void build_one_s_matrix(int alpha, int beta, Mat G_perm_ab,
                            PetscInt perm_rstart, PetscInt perm_rend,
                            PetscInt perm_cstart, PetscInt perm_cend);
    void build_scatters(Vec s_proto, Vec t_proto);
};

} // namespace tndm
```

### Phase 3: Sub-Matrix Extraction, Permutation, and Matrix-Free Construction

The construction pipeline for each `(α,β)` pair reuses steps from `HMatrixGreenFunction`:

1. `extract_component_submatrix(G_dense, α, β)` → `G_αβ` (`Np×Np` dense, original DOF order).
2. Apply the same PCA-Hilbert spatial permutation → `G_αβ_perm`.
3. Build the matching cluster tree from PETSc's ownership ranges of `G_αβ_perm`.
4. Pre-allocate `x_work` and `y_work` Vecs with `VecCreateMPIWithArray(..., nullptr, ...)`.
5. Construct `S_ab_[idx]` via `construct_matrix_free(MPIComm(comm_), Np, Np, Amult, opts, &row_tree)` where the callback uses `VecPlaceArray`/`VecResetArray` (zero-copy, see §3.3).
6. Destroy `x_work`, `y_work`.

```cpp
strumpack::structured::StructuredOptions<double> opts(
    strumpack::structured::Type::BUTTERFLY);
opts.set_rel_tol(config_.rtol);
opts.set_leaf_size(config_.leaf_size);
opts.set_max_rank(config_.max_rank);

ClusterTree row_tree = build_petsc_tree(dist, 0, n_ranks);  // see §3.2

S_ab_[idx] = construct_matrix_free<double>(
    strumpack::MPIComm(comm_), Np, Np, Amult, opts, &row_tree);
```

**Notes:**
- The Hilbert permutation is applied to `G_αβ` before building the STRUMPACK matrix. The cluster tree is built on the already-permuted matrix — its `n` corresponds to permuted row indices, not physical coordinates.
- `planar_fault=true`: skip `G_0β` components (leave `S_ab_[0*slip_D_+β] = nullptr`), identical to the HTool path.
- Scatters (`scatter_s_`, `scatter_t_`) are built exactly as in `HMatrixGreenFunction::build_scatters`, using the `StructuredMatrix::dist()` to determine the local row/col ranges.

### Phase 4: Config Mapping

| `HMatrixConfig` field | STRUMPACK setter              |
|-----------------------|-------------------------------|
| `rtol`                | `opts.set_rel_tol(...)`       |
| `leaf_size`           | `opts.set_leaf_size(...)`     |
| `max_rank`            | `opts.set_max_rank(...)`      |

### Phase 5: Benchmark App

Create `app/gf-strumpack-bench.cpp` (thin wrapper around `StrumpackGFOperator`):

1. Load the dense GF checkpoint (same path as `gf-hmatrix-validate`).
2. Construct `StrumpackGFOperator` — triggers matrix-free construction of all `D*(D-1)` sub-matrices.
3. Accuracy check: compare $\|S \cdot x - G_{dense} \cdot x\| / \|G_{dense} \cdot x\|$ for random $x$.
4. Print per-component and aggregate results:

```
STRUMPACK BUTTERFLY construction summary (D=2, 2 sub-matrices, Np=1400)
  Sub-matrix  Forward MatVecs  Adjoint MatVecs  Compression  Rel. error
  G_00        Nf_00            Na_00            XX.X %       Y.YYe-ZZ
  G_10        Nf_10            Na_10            XX.X %       Y.YYe-ZZ
  -----------------------------------------------------------------------
  TOTAL       Nf_total         Na_total
  Projected PDE-solver evaluations (if forward-only): Nf_total
```

The "Projected PDE-solver evaluations" line is the primary output: the number of full PDE solves that would be needed if each forward MatVec were replaced by a real solver call.

### Phase 6: Wire into `SeasQDDiscreteGreenOperator`

After the benchmark validates accuracy and MatVec count, wire `StrumpackGFOperator` into `SeasQDDiscreteGreenOperator` as an optional backend alongside the existing HTool path:

* Add `use_strumpack` flag to `HMatrixConfig`.
* In `build_h_matrix()`: branch on `use_strumpack` to construct `StrumpackGFOperator` instead of `HMatrixGreenFunction`.
* Route `update_traction()` through `strumpack_op_->apply()` when active.

---

## 5. Checklist

### Branch: `ymagen/strumpack-bench`

- [ ] Update `STRUMPACK_TRANSITION_PLAN.md` with bridge design details ← **this document**
- [ ] Add `find_package(STRUMPACK)` and new target to `app/CMakeLists.txt`
- [ ] Implement `StrumpackGFOperator` with `BUTTERFLY` type:
  - [ ] `build_petsc_tree()` — cluster tree matching PETSc's PETSC_DECIDE layout
  - [ ] `build_one_s_matrix()` — zero-copy `mult_1d_t` callback with `VecPlaceArray`
  - [ ] `build_scatters()` — reuse from `HMatrixGreenFunction`
  - [ ] `apply()` — reuse from `HMatrixGreenFunction`
- [ ] Create `gf-strumpack-bench.cpp` and confirm it compiles and links
- [ ] Run on BP1: record per-component MatVec counts, error, and memory vs HTool (2 sub-matrices 1400×1400 for 2D)
- [ ] Wire `StrumpackGFOperator` into `SeasQDDiscreteGreenOperator` (Phase 6, after benchmark validates)
