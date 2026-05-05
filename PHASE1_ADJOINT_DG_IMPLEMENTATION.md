# Phase 1: Adjoint DG Implementation Plan

This document outlines the implementation plan for Phase 1 of the Green's function ($G$) H-matrix integration. The primary goal is to build and verify the adjoint of the Discontinuous Galerkin (DG) operator $A$, which is a prerequisite for computing the adjoint of the Green's function matrix $G^T$.

## 1. Mathematical Derivation: Discrete Adjoint of the DG Operator

The forward DG elasticity operator $A$ maps displacement $u$ to a residual $r$. In a matrix-free context, $y = Ax$ is computed by evaluating volume and facet integrals.

To compute the adjoint action $y = A^T x$, we must formulate the discrete adjoint. For a bilinear form $a(u, v)$ such that $(Au, v) = a(u, v)$, the adjoint operator $A^*$ is defined by $(Au, v) = (u, A^* v)$. In the discrete setting, this corresponds to the transpose of the matrix $A$.

### Symmetry and Non-Symmetry
- **SIPG (Symmetric Interior Penalty Galerkin):** The SIPG formulation is symmetric by design. However, non-symmetry arises from:
    - **Boundary Conditions:** Certain boundary conditions (e.g., specific traction or displacement couplings) can break symmetry.
    - **Fault Terms:** The mapping from slip on the fault to traction involves terms that are not inherently symmetric when viewed as part of the full domain operator.

### Transpose Action
For each block $A_{ij}$ representing the influence of element $j$ on element $i$, the adjoint action requires applying $A_{ij}^T$.
- **Forward:** $y_i = \sum_j A_{ij} x_j$
- **Transpose:** $y_i = \sum_j A_{ji}^T x_j$

In Tandem's element-centric `apply` logic, this means that when processing element $i$, we must gather contributions that element $i$ would have made to its neighbors in the forward pass.

## 2. Proposed Changes

### 2.1. Core Library (`src/`)

| File | Change Description |
| :--- | :--- |
| `src/form/AbstractDGOperator.h` | Add `virtual void apply_transpose(BlockVector const& x, BlockVector& y) = 0;` to the interface. |
| `src/form/DGOperator.h` | Implement `apply_transpose`. This will require a "reverse" scatter/accumulate pattern to handle contributions to ghost elements. |

### 2.2. Application Logic (`app/`)

| File | Change Description |
| :--- | :--- |
| `app/localoperator/Elasticity.h` | Add `apply_transpose` to the `Elasticity` local operator. |
| `app/localoperator/Elasticity.cpp` | Implement `apply_transpose` by utilizing transposed Yateto kernels. Specifically, transpose the volume and facet flux kernels to compute $A_{ji}^T x_j$. |
| `app/common/PetscDGShell.h` | Declare `apply_transpose` static method for PETSc callback. |
| `app/common/PetscDGShell.cpp` | Implement `PetscDGShell::apply_transpose` and register it using `MatShellSetOperation(A_, MATOP_MULT_TRANSPOSE, ...)` in the constructor. |
| `app/common/PetscLinearSolver.h` | Add `solve_transpose()` method. |
| `app/common/PetscLinearSolver.cpp` | Implement `solve_transpose()` using `KSPSolveTranspose(ksp_, b_->vec(), x_->vec())`. |

### 2.3. Green's Function Operator

| File | Change Description |
| :--- | :--- |
| `app/form/SeasQDDiscreteGreenOperator.h` | Add `apply_transpose(Vec x, Vec y)` to support $G^T$. |
| `app/form/SeasQDDiscreteGreenOperator.cpp` | Implement $G^T y = B^T (A^T)^{-1} T^T y$. This requires adjoint versions of the Traction ($T$) and Slip-to-RHS ($B$) operators. |

### 2.4. Adjoint Adapter Operators ($T^T$ and $B^T$)

To compute the adjoint Green's function action $G^T y$, we must implement the adjoints of the operators that couple the fault to the domain.

| File | Change Description |
| :--- | :--- |
| `app/form/AbstractAdapterOperator.h` | Add `virtual void apply_traction_adjoint(BlockVector const& traction, BlockVector& rhs) = 0;` to the interface. |
| `app/form/AdapterOperator.h` | Implement `apply_traction_adjoint` by integrating the input traction coefficients against the DG test functions on the fault, effectively performing the adjoint of the traction evaluation. |
| `app/localoperator/Adapter.h` | Add `traction_adjoint` and `slip_adjoint` methods to the `Adapter` class. |
| `app/localoperator/ElasticityAdapter.cpp` | Implement `traction_adjoint` and `slip_adjoint` using transposed Yateto kernels. |

#### Step-by-Step Implementation of $G^T y = B^T (A^T)^{-1} T^T y$:

1.  **Map Traction to RHS ($T^T$):**
    *   Input: A vector $y$ of traction coefficients on the fault.
    *   Action: Evaluate $y$ at quadrature points and integrate against the "traction-test" terms of the DG formulation. This is mathematically equivalent to the adjoint of the displacement-to-traction mapping.
    *   Result: A DG Right-Hand Side vector $b_{adj}$.

2.  **Adjoint DG Solve ($(A^T)^{-1}$):**
    *   Action: Solve the adjoint DG system $A^T v = b_{adj}$.
    *   Tool: Use `PetscLinearSolver::solve_transpose()`.
    *   Result: An adjoint displacement field $v$.

3.  **Evaluate Adjoint Slip ($B^T$):**
    *   Action: Evaluate the "traction-like" dual quantity of the adjoint solution $v$ on the fault. This involves evaluating stress and penalty terms of $v$ and integrating them against the fault basis functions.
    *   Result: The final adjoint slip vector $r = G^T y$.

## 3. New Files to be Added

| File | Purpose |
| :--- | :--- |
| `test/adjoint_consistency.cpp` | A dedicated unit test to verify the adjoint identity $\langle Ax, y \rangle = \langle x, A^T y \rangle$ and $\langle Gx, y \rangle = \langle x, G^T y \rangle$ to machine precision. |

## 4. The Adjoint Consistency Test (Critical)

The test will perform the following steps:
1. Initialize the DG operator $A$ (and Green's function $G$ if applicable).
2. Generate two random PETSc vectors $x$ and $y$ of appropriate sizes.
3. Compute $v = Ax$ and then the inner product $I_1 = \langle v, y \rangle$.
4. Compute $w = A^T y$ and then the inner product $I_2 = \langle x, w \rangle$.
The Adjoint Consistency Test (Critical): Before proceeding, you must verify the adjoint to machine precision using random PETSc vectors $x$ and $y$. You must assert that the inner product matches:
$$\langle Gx, y \rangle - \langle x, G^T y \rangle < \epsilon_{\text{machine}}$$
Do not move past this step until this identity holds. If the adjoint is even slightly algebraically inconsistent, the randomized peeling will produce garbage off-diagonal blocks.

**Success Criterion:** This identity must hold for both the matrix-free `PetscDGShell` and the assembled `PetscDGMatrix` (if used). Failure in this test indicates an algebraic error in the transposed kernel implementation or boundary condition handling.
