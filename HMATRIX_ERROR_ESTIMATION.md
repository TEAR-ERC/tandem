# H-Matrix Validation Error Estimation

## Observed Symptoms
1. **Kernel-based approach (`MatCreateH2OpusFromKernel`):** High relative error (~1.76).
2. **Matrix-based approach (`MatCreateH2OpusFromMat`):** Termination with `Assertion hmatrix.sym == true failed` in H2Opus (`hlru_sym_template`).
3. **Common context:** Square H-matrix blocks (1400x1400) constructed on identical row and column point clouds.

## 1. Why the Kernel-based Approach Fails
The high relative error (1.76) suggests the H-matrix is essentially a "miss" compared to the dense Green's Function. 

*   **Interpolation vs. Discrete Lookup:** `MatCreateH2OpusFromKernel` (with `basisord > 0`) uses **Chebyshev interpolation** to build low-rank approximations of off-diagonal blocks. This requires evaluating the kernel function at Chebyshev nodes—points in space that do not coincide with our physical fault nodes.
*   **The "Dirac" Kernel Problem:** Our `GFKernel` is a discrete lookup table. It returns 0.0 for any coordinate triple not exactly present in our `coord_to_idx_` map. Since Chebyshev nodes are almost never exact physical node coordinates, the H2Opus construction sees a kernel that is zero almost everywhere.
*   **Brittleness:** Even if H2Opus attempted to evaluate only at the provided points, `std::map` lookup on floating-point coordinates is extremely brittle. Any infinitesimal precision loss in H2Opus's internal tree traversal would result in a lookup failure.

**Conclusion:** The kernel-based path is fundamentally incompatible with a dense-matrix lookup "kernel".

## 2. Why the Matrix-based Approach Hits an Assertion
`MatCreateH2OpusFromMat` uses **Randomized Adaptive Approximation (HARA)**, which treats the input matrix as a black-box operator. This is the correct path for compressing a pre-assembled dense matrix.

*   **Symmetry Assumption:** The error `Assertion hmatrix.sym == true failed` occurs in `hlru_sym_template`. This implies H2Opus has decided that the matrix should be symmetric.
*   **Trigger:** In H2Opus, if a square matrix is constructed using the same point cloud for both rows and columns, the internal cluster tree is shared. H2Opus (or the PETSc wrapper) likely defaults to a symmetric H-matrix structure in this case.
*   **Conflict:** Green's Functions in Tandem are generally **non-symmetric** (e.g., $G_{traction, slip} \neq G_{slip, traction}$), especially when the traction components and slip components represent different physical quantities or directions. 
*   **Failure to Force Non-Symmetry:** Despite setting `-mat_h2opus_sym false` and `MAT_SYMMETRIC = PETSC_FALSE`, the randomized construction path in H2Opus seems to be bypassing these flags and asserting symmetry based on the geometric identity of the row/column points.

## 3. Proposed Path Forward
To resolve this, we must successfully use `MatCreateH2OpusFromMat` while forcing a "General" (non-symmetric) square H-matrix.

1.  **Diagnostic Check:** Verify if H2Opus requires distinct point clouds (even if coordinates are identical) to treat a square matrix as non-symmetric. 
2.  **PETSc API Investigation:** Identify if there is a specific `MatSetOption` or H2Opus-specific flag (e.g., in `H2OpusHandle`) that prevents the transition to the symmetric construction path.
3.  **Workaround:** If H2Opus refuses to build a non-symmetric square matrix on a single tree, we can provide two "different" point clouds to `MatCreateH2OpusFromMat` by adding a negligible epsilon to one of them, forcing H2Opus to build two separate trees and a general H-matrix.

## 4. Comparison with `htool` (MATHTOOL)
Using `htool` instead of `h2opus` is highly likely to resolve both the accuracy and the symmetry issues.

### Index-based Kernel (Solves Accuracy)
Unlike `h2opus`, which passes **coordinates** to the kernel, `htool` passes **indices**:
```cpp
PetscErrorCode GenEntries(PetscInt sdim, PetscInt M, PetscInt N, 
                          const PetscInt *J, const PetscInt *K, 
                          PetscScalar *ptr, PetscCtx ctx)
```
*   **Direct Lookup:** We can use the indices `J[j]` and `K[k]` to index directly into our dense Green's Function. 
*   **No Interpolation:** `htool` typically uses ACA (Adaptive Cross Approximation) which evaluates the kernel at specific discrete pairs. This avoids the Chebyshev interpolation problem where `h2opus` evaluates at "empty" spatial locations.
*   **Zero Precision Loss:** Since we index by integer, there is no floating-point mapping or map-lookup fragility.

### Explicit Symmetry Control (Solves Assertion)
The `MATHTOOL` interface in PETSc explicitly supports non-symmetric matrices even when row and column coordinates are identical:
```cpp
PetscCall(MatCreateHtoolFromKernel(..., coords, coords, kernel, ...));
PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_FALSE));
```
Because `htool` is a different library with a different underlying structure (often optimized for Boundary Element Methods which are non-symmetric), it does not share the rigid symmetry assertions observed in H2Opus's randomized path.

**Conclusion:** `htool` is architecturally a better fit for compressing an existing dense Green's Function in Tandem.
