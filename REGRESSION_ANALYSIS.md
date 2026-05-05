# Regression Analysis: `ymagen/h_matrix_h2opus` vs `origin/dmay/staging`

## Observed behavior

| Branch | Steps to first co-seismic event | Step 100 physical time |
|--------|--------------------------------|------------------------|
| `origin/dmay/staging` | ~20 000 steps | ~70 M s (~2.2 yr) |
| `ymagen/h_matrix_h2opus` (`use_hmatrix=false`) | >400 000 steps | ~1.28 M s (~15 days) |

Both GF mode and H-matrix mode show the same wrong behavior in the current branch — confirming that the bug lies upstream of the H/G split, i.e., in G_ itself or in the time-stepping state it produces.

The two branches diverge at **step 4** (first adaptive-dt choice):
- staging: dt = 648.4 s
- current: dt = 636.2 s (~2 % lower)

By step 30 the ratio has grown to ~33 %, and the systems are in qualitatively different physical phases.

---

## Code changes between branches (from `git diff origin/dmay/staging`)

The following files were modified. All changes are **strictly additive** with respect to the `use_hmatrix=false` code path:

| File | Change relevant to non-H-matrix path |
|------|--------------------------------------|
| `AbstractFrictionOperator.h` | Added two pure-virtual methods (`fill_fault_node_coords`, `fault_num_basis_functions`) — interface only |
| `FrictionOperator.h` | Implemented the two new virtuals by delegating to `lop_` — additive |
| `RateAndStateBase.h` | Added `cl()`, `fault_node_coords()`, `fill_fault_node_coords()` — additive |
| `SeasConfig.h/cpp` | Added `HMatrixConfig hmatrix_config` to `Config` and its TOML schema — additive, defaults to `use_hmatrix=false` |
| `SEAS.cpp` | Passes `cfg.hmatrix_config` to `SeasQDDiscreteGreenOperator` ctor; added `validateGFHMatrix()` |
| `SeasQDDiscreteGreenOperator.h/cpp` | Added H-matrix members, `compute_fault_coordinates`, `build_h_matrix`, `validate_all` — all guarded by `#ifdef PETSC_HAVE_HTOOL` and `if (hmatrix_config_.use_hmatrix)` |
| `CMakeLists.txt` | Added new `gf-hmatrix-validate` executable target — no effect on `tandem` target |

**Conclusion from code inspection:** The `use_hmatrix=false` code path through `update_traction → MatMult(G_, …)` is **byte-for-byte identical** between the two branches. The `GreensFunctionIndices` constructor, `partial_assemble_discrete_greens_function`, `compute_boundary_traction`, and `update_traction` are unchanged.

---

## Pre-existing bug (same in both branches — not a new source of divergence)

In `partial_assemble_discrete_greens_function` (`SeasQDDiscreteGreenOperator.cpp:545`):

```cpp
if (i >= ind.nb_offset && i < ind.nb_offset + ind.m) {  // BUG: ind.m, should be ind.n
    VecSetValue(S_->vec(), i, 1.0, INSERT_VALUES);
}
```

For 2D BP1 elasticity, the kernel tensors give:
- `traction_block_size = traction::Shape[0] * traction::Shape[1] = 7 * 2 = 14` (7 basis functions × 2 traction components)
- `slip_block_size = nbf * TangentialComponents = 7 * 1 = 7`
- Therefore `m = 2 * n`

The condition should check `i < ind.nb_offset + ind.n` (slip DOF range), not `ind.m` (traction DOF range). For a **single-process run**, `ind.nb_offset = 0` and `i < 0 + 2n` is always satisfied for `i ∈ [0, n-1]`, so the bug is invisible. For **multi-process MPI**, a process can call `VecSetValue` for column indices that belong to the next process — with `INSERT_VALUES` and value 1.0, PETSc's assembly phase should route this correctly, but it is formally incorrect and could cause assertion failures in PETSc debug builds.

Since the identical bug is present in staging, this cannot explain the observed performance regression.

---

## Hypotheses for the regression (in descending likelihood)

### H1 — Stale GF checkpoint loaded by current branch

**What happens:** If the user has `[gf_checkpoint]` in their TOML and a `gf_mat.bin` from a prior staging run exists at that path, the current branch loads it. If the checkpoint was created with a different MPI commsize, `repartition_gfs_ = true` and `G_` is permuted via `MatMatMult(Rperm, G_*Cperm, ...)`. Even if commsize matches, loading a checkpoint from a different code version can produce a G_ that differs in memory layout from a freshly computed one (PETSc binary format stores metadata that may differ across versions). The resulting G_ silently gives wrong traction values, leading to wrong slip dynamics — **with the same magnitude and direction of bias for both GF and H modes**, since H_ is derived from G_.

**How to validate:**
1. Delete (or rename) `gf_mat.bin`, `gf_vec.bin`, `gf_facet_labels.bin` at the checkpoint path.
2. Re-run with current branch. If the dynamics now match staging, this was the cause.
3. Alternatively, temporarily set `checkpoint_every_nmins` to `1e10` and remove the `[gf_checkpoint]` section from the TOML.

---

### H2 — PETSc recompiled with HTools support changes BLAS linkage or solver behavior

**What happens:** If PETSc was recompiled between when staging was tested and the current branch was built (e.g., to enable HTools via `--download-htool` or `--with-htool-dir`), the resulting PETSc might link to a different BLAS implementation or use different dense matrix routines. The GF assembly calls `base::solve(0.0, S_view)` for each of the ~1400 columns — this is an iterative solve (GMRES/CG + AMG). A different BLAS library (e.g., OpenBLAS vs MKL, or different vectorization flags) changes the rounding in DGEMV/DDOT, causing the iterative solver to take different paths to convergence. After 1400 × (many GMRES iterations) FMA operations, the resulting G_ can differ at the 1e-10 to 1e-5 level — sufficient to shift the adaptive time step by 2 % at step 4 and produce qualitatively different long-time dynamics.

**How to validate:**
1. `grep PETSC_HAVE_HTOOL $(petscconf --petscdir)/include/petscconf.h` in both build environments.
2. `ldd $(which tandem)` or `ldd build/app/tandem` and compare the BLAS/LAPACK `.so` names between staging and current builds.
3. If possible, rebuild both branches against the **same** PETSc installation without HTool, then compare time-step sequences.

---

### H3 — `PETSC_HAVE_HTOOL` changes PETSc global state at init time

**What happens:** When `PETSC_HAVE_HTOOL` is defined, the `tandem` binary now includes `<petscmathtool.h>`. PETSc headers can trigger static-initializer registration of matrix types and operations. If including this header registers a new `MATHTOOLDENSE` or similar matrix type, PETSc may internally change the default for `MatCreateDense` (e.g., picking a different implementation or changing memory alignment). This would affect G_ even before the H-matrix path is entered.

**How to validate:**
1. Compile the current branch without HTools in PETSc (`-DPETSC_HAVE_HTOOL` not defined).
2. Add a `PetscPrintf` to print `MatGetType(G_)` after `MatCreateDense` and compare across builds.
3. Run `PetscOptionsView(NULL)` at startup to see what options PETSc registers — compare between staging and current binary.

---

### H4 — Partial recompile / stale object files

**What happens:** If the build directory was not fully cleaned between branch switch, some `.o` files from staging remain. The linker might combine old code (staging's `SeasQDDiscreteGreenOperator.o`) with new code (`SEAS.o` from the current branch), producing an internally inconsistent binary. This is particularly likely if only the changed translation units were recompiled.

**How to validate:**
1. `cd build && make clean && cmake .. -D... && make -j4 tandem`
2. Re-run BP1. If dynamics match staging, partial recompile was the cause.

---

### H5 — TOML configuration change

**What happens:** When adding the `[hmatrix]` section to the BP1 TOML, another parameter (mesh resolution, final_time, cfl, etc.) was accidentally changed. Even a 1 % change in mesh resolution changes the GF matrix.

**How to validate:**
1. `diff <staging-toml> <current-toml>` — compare the actual TOML files used in both runs.
2. Check that `resolution`, `cfl`, `lib`, `scenario`, `ref_normal`, `up` are identical.

---

### H6 — `MatSetValuesBlocked` column indexing for multi-rank runs (new scenario)

**What happens:** If the user recently switched from single-rank to multi-rank runs (e.g., to test H-matrix performance), the pre-existing `ind.m` bug becomes active. As shown above: with 2 MPI processes and m=2n, process 0 sets `S_[n0] = 1.0` for some columns that belong to process 1 **in addition to** process 1 doing the same. More critically, for columns `i ∈ [n0, 2*n0)` (an "extended" range of process 0), only process 0 calls `VecSetValue` — but those DOFs may not exist in the global DOF range `[0, n_total)` at all. If `n0 + m0 > n_total`, some `VecSetValue` calls use out-of-bounds indices, and PETSc's behavior is undefined (silent wrong values or error depending on build type).

**How to validate:**
1. Verify: is the user running with `mpirun -n 1` (single process) or multiple ranks?
2. If multiple ranks: add an assertion `assert(ind.nb_offset + ind.n <= n_gf)` and `assert(ind.nb_offset + ind.m <= n_gf)` to expose the bug.
3. Fix: change `ind.nb_offset + ind.m` → `ind.nb_offset + ind.n` on line 545.

---

## Recommended validation sequence

Run these in order — stop at whichever one isolates the cause:

```bash
# 1. Confirm no checkpoint contamination
rm -f <checkpoint_path>/gf_mat.bin <checkpoint_path>/gf_vec.bin <checkpoint_path>/gf_facet_labels.bin
./tandem bp1.toml  # re-run current branch fresh

# 2. Check TOML is unchanged
diff bp1_staging.toml bp1_current.toml

# 3. Full clean rebuild
cd build && make clean && cmake .. && make -j4

# 4. Compare PETSc builds
grep PETSC_HAVE_HTOOL $(pkg-config --variable=includedir PETSc)/petscconf.h
ldd app/tandem | grep -E "blas|lapack|mkl"

# 5. Print G_ norm after assembly (add temporarily to get_discrete_greens_function)
# PetscPrintf(base::comm(), "||G_||_F = %g\n", (double)norm);  // via MatNorm(G_, NORM_FROBENIUS, &norm)
# Compare the printed value between staging and current for same input.

# 6. Confirm single-rank vs multi-rank
mpirun -n 1 ./tandem bp1.toml   # forces single process
```

## The one-line fix for the known pre-existing bug

```cpp
// SeasQDDiscreteGreenOperator.cpp:545 — change ind.m to ind.n
if (i >= ind.nb_offset && i < ind.nb_offset + ind.n) {  // was ind.m (= 2*n for 2D elasticity)
```

This is safe to apply regardless of the current investigation — it is correct for all cases (single-process and multi-process, Poisson and Elasticity) and has no effect on single-process runs.
