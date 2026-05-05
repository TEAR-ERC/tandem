# Stage 1: Pragmatic Green's Function H-Matrix Implementation

This document outlines the first stage of implementing an H-matrix representation for the Green's function ($G$). This approach focuses on validation and isolation of the H-matrix logic by utilizing the existing dense assembly pipeline.

## 1. Extracting Fault Geometry (The Point Cloud)

H2Opus requires physical coordinates for all degrees of freedom (DOFs) on the fault to build spatial cluster trees. Tandem is dimension-agnostic and can run in 2D or 3D, controlled by the `DomainDimension` constant (defined in `app/config.h`).

### Implementation Strategy
*   **Target:** `SeasQDDiscreteGreenOperator` class.
*   **Dimension Handling:** Set `spacedim = DomainDimension`.
*   **Coordinate Extraction Logic:**
    1.  **Access Components:** Retrieve the `Curvilinear<DomainDimension>` map (`cl_`) and the `Adapter`.
    2.  **Identify Nodes:** For the fault facets, Tandem uses a `NodalRefElement<DomainDimension - 1u>`. Retrieve its reference nodes via `refElement.refNodes()`.
    3.  **Map to Physical Space:**
        *   Iterate over all local fault facets using the `fault_map`.
        *   For each facet `faceNo`, and each reference node `chi` in the basis:
            *   Call `cl_->facetParam(faceNo, chi)` to get the physical coordinate `x` (a `std::array<double, DomainDimension>`).
            *   In 2D, `x` will contain $(x, y)$. In 3D, it will contain $(x, y, z)$.
    4.  **DOF Alignment:** Since the GF matrix $G$ is blocked by facet (using `MatSetValuesBlocked`), we must provide coordinates for every row/column in the matrix. 
        *   If the basis is nodal, each DOF corresponds exactly to one of the extracted points.
        *   If multiple components (e.g., $u_x, u_y, u_z$) share the same node, the coordinates for those rows/columns in the `coords` array passed to H2Opus will be identical.
*   **Storage:** The final `coords` array for `MatCreateH2OpusFromMat` will have a total of $N_{rows} \times DomainDimension$ entries.

## 2. Dense Green's Function Assembly

We will leverage the existing `SeasQDDiscreteGreenOperator::create_discrete_greens_function` and `partial_assemble_discrete_greens_function` routines.

### Key Requirements
*   Ensure the matrix $G$ is created as a `MATDENSE` (standard in current implementation).
*   Complete the full assembly of $G$ before attempting H-matrix construction.

## 3. H-Matrix Construction via MatCreateH2OpusFromKernel

To support parallel execution, we will use **`MatCreateH2OpusFromKernel`**. This allows us to construct the H-matrix in a distributed environment by providing a kernel function that evaluates entries on-the-fly. In Stage 1, this kernel will act as a "dummy" wrapper that looks up values from our pre-assembled dense Green's Function.

### The `MatCreateH2OpusFromKernel` Interface

```cpp
PetscErrorCode MatCreateH2OpusFromKernel(
    MPI_Comm comm,      // MPI communicator
    PetscInt m,         // Local rows
    PetscInt n,         // Local columns
    PetscInt M,         // Global rows
    PetscInt N,         // Global columns
    PetscInt spacedim,  // DomainDimension (2 or 3)
    const PetscReal coords[], // Fault DOF coordinates
    PetscBool cdist,    // Whether coordinates are distributed
    MatH2OpusKernelFn *kernel, // The dummy lookup kernel
    void *kernelctx,    // Context containing the dense matrix G
    PetscReal eta,      // Admissibility (e.g., 0.6 - 1.0)
    PetscInt leafsize,  // Cluster tree leaf size (e.g., 32)
    PetscInt basisord,  // Chebychev approximation order (e.g., 8)
    Mat *nA             // Output H-matrix
)
```

### The Dummy Lookup Kernel

Since `MatCreateH2OpusFromKernel` passes physical coordinates to the kernel function rather than indices, we must map them back to matrix indices for the dense lookup. 

**Recommended Mapping Approach:**
During the coordinate extraction phase (Step 1), build a spatial index (e.g., `std::unordered_map<std::vector<double>, PetscInt>`) that maps the `spacedim` coordinates back to the DOF index.

```cpp
PetscScalar GFKernel(PetscInt sdim, PetscReal x[], PetscReal y[], void *ctx) {
    auto* kernelCtx = static_cast<MyKernelCtx*>(ctx);

    // 1. Map physical coordinates x[] and y[] to indices i and j.
    // Use the spatial index built during Step 1.
    PetscInt i = kernelCtx->lookupIndex(x);
    PetscInt j = kernelCtx->lookupIndex(y);

    // 2. Return value from the pre-computed dense matrix
    PetscScalar val;
    // Note: If denseG is distributed, use MatGetValues and handle global indexing
    MatGetValues(kernelCtx->denseG, 1, &i, 1, &j, &val);
    return val;
}
```

### Tuning Parameters and Options
*   **`eta` (Admissibility):** Controls the balance between compression and accuracy. Typical values are $0.6$ to $1.0$.
*   **`leafsize`:** The number of DOFs in the smallest cluster blocks. Standard starting point is $32$.
*   **`basisord`:** The Chebychev approximation order. Higher values increase accuracy at the cost of memory and computation. Typical range is $4$ to $12$.
*   **Options Database:**
    *   `-mat_h2opus_leafsize <PetscInt>`
    *   `-mat_h2opus_eta <PetscReal>`
    *   `-mat_h2opus_order <PetscInt>` (corresponds to `basisord`)
    *   `-mat_h2opus_normsamples <PetscInt>`

### Advantages for Parallelism
*   **Scalability:** Unlike `MatCreateH2OpusFromMat`, this method is fully supported in parallel.
*   **Distributed Clustering:** H2Opus will handle the distributed construction of cluster trees and block partitions across the MPI communicator.



## 4. Validation Routine

To ensure correctness, we must verify the H-matrix against the original dense GF.

### Steps
1.  Generate a random vector $x$ using `VecSetRandom`.
2.  Compute $y_{dense} = G_{dense} x$ using `MatMult`.
3.  Compute $y_{H} = H_{opus} x$ using `MatMult`.
4.  Calculate the relative error:
    $$\frac{||y_{dense} - y_{H}||}{||y_{dense}||}$$
5.  **Success Criterion:** The error should be within the order of the compression tolerance $\epsilon$.

## 5. Proposed Code Changes

| Component | Task |
| :--- | :--- |
| `app/form/SeasQDDiscreteGreenOperator` | Add `get_fault_coordinates()` method to extract DOF positions. |
| `app/form/SeasQDDiscreteGreenOperator` | Add `build_h_matrix()` method that invokes the H2Opus constructor after dense assembly. |
| `test/h_matrix_validation.cpp` | Create a new test case to perform the coordinate extraction and MV product validation. |
