# Randomized Linear Algebra in Tandem

Tandem implements randomized subspace methods to compute truncated decompositions of large operators. These methods are primarily used for eigenvalue deflation in preconditioners.

## Implementation Location
- **Files:** `app/pc/reig_aux.h` and `app/pc/reig_aux.c`
- **Key Functions:**
    - `RandEigsMax`: Computes the $k$ largest eigenvalues/eigenvectors.
    - `RandEigsMin`: Computes the $k$ smallest eigenvalues/eigenvectors using a shift-and-invert strategy.

## Algorithm: Randomized Subspace Iteration
The implementation follows the randomized range-finding framework (Halko et al., 2011):

1. **Random Probing:** A random dense matrix is generated using `MatSetRandom`.
2. **Subspace Power Iteration:** The operator is applied multiple times (controlled by `power_its`) to the random matrix to enhance the projection of the dominant singular/eigen-space.
3. **Orthonormalization:** `VecOrthonormalize_ModifiedGramSchmidt` is used to create an orthonormal basis for the discovered subspace.
4. **Projection (Rayleigh-Ritz):** The high-dimensional operator is projected onto the subspace using `MatPtAP` (or `MatPtInvAP` for the "Min" variant).
5. **Small Dense Solve:** The resulting small dense matrix is solved using LAPACK's `dsyev` via the `lapackeigs` helper.
6. **Recovery:** The eigenvectors are mapped back to the global space.

## Comparison to Randomized SVD
While the functions are named for **Eigendecomposition**, the algorithm used for the range-finding step is identical to that used in a **Randomized SVD (RSVD)**. For a symmetric operator $A$, the randomized eigendecomposition and RSVD are mathematically equivalent. In the context of `tandem`, these methods provide the low-rank approximation necessary for efficient preconditioning of the DG systems.

## Parameters
- `k`: Target rank (number of eigenvalues).
- `o`: Oversampling parameter (additional random vectors to improve accuracy).
- `pits`: Number of power iterations.
- `prand`: PETSc random context.
