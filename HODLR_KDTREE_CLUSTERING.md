# HODLR Compression Failure on 3D Faults: Diagnosis and Fix

## Background

The discrete Green's function (GF) for earthquake cycle simulations maps fault
slip to fault traction. For an N-node fault discretisation with D traction
components and `slip_D = D-1` slip components, the GF is a dense matrix of size
`D·Np × slip_D·Np` where `Np = N_el · nbf` is the total number of scalar
fault nodes.

We compress the GF with STRUMPACK's HODLR (Hierarchically Off-Diagonal
Low-Rank) format, using ButterflyPACK as the backend.  The key property HODLR
exploits is that **off-diagonal blocks between well-separated spatial clusters
are numerically low-rank**.

---

## Observed Failure on bp7 (3D, 2D fault)

### Configuration

| Parameter | Value |
|---|---|
| Problem | BP7, 3D elasticity |
| Matrix size | 16560 × 11040 (D=3, Np=5520) |
| rtol | 1e-4 |
| leaf\_size | 64 (later tried 512) |
| max\_rank | 64 (later tried 512) |

### Symptom

HODLR construction reports errors > 1.0 at fine hierarchical levels, and the
final accuracy `‖Sv - Gv‖ / ‖Gv‖ ≈ 5.16` — more than four orders of magnitude
above the requested tolerance.

```
Level 1  rank=275  error=4.9e-4  ← converges (after 3 trials)
Level 2  rank=163  error=7.8e-4  ← OK
Level 3  rank=180  error=7.9e-4  ← OK
Level 4  rank=115  error=1.1e-3  ← OK
Level 5  rank= 96  error=6.07    ← BLOWN UP
Level 6  rank= 86  error=2.45    ← BLOWN UP  (rank ≈ N_leaf = block size)
Level 7  fullmat done
```

Increasing `leaf_size` and `max_rank` to 512 had **no effect**: the memory
footprint and error were essentially unchanged.  The reason is explained below.

---

## Root Cause 1: leaf\_size is Ignored

Our `build_petsc_tree` constructs a ClusterTree from PETSc's MPI rank
distribution.  With 30 ranks and Np=5520, this tree has **30 leaves of size
≈ 184 each**.

When an explicit ClusterTree is passed to `construct_matrix_free`, ButterflyPACK
uses that tree as the top-level skeleton but **further subdivides each rank-level
leaf based on its own internal leaf size** (defaulting to ≈ 64).  The
`StructuredOptions::set_leaf_size(512)` setting is a STRUMPACK parameter that
does not propagate into ButterflyPACK's internal subdivision.

**Evidence**: `N_leaf` remained at 86 (= ceil(5520/64)) regardless of the
configured leaf\_size.  Memory and MatVec counts were virtually unchanged across
both runs.

---

## Root Cause 2: 1D Hilbert Ordering on a 2D Fault

This is the fundamental, physics-driven problem.

### 1D fault (bp3, 2D problem) — HODLR works

The fault is a line segment.  The Hilbert-ordered nodes form a 1D sequence.
Splitting this sequence at the midpoint gives a **left half** and a **right
half** that are spatially separated along the fault.

```
Fault nodes in order:  ● ● ● ● ● ● ● ● | ● ● ● ● ● ● ● ●
                        ←── cluster A ──→ ←── cluster B ──→
```

Off-diagonal blocks between A and B correspond to purely far-field interactions
→ **genuinely low rank** (rank 6–7 in practice).

### 2D fault (bp7, 3D problem) — HODLR breaks

The fault is a 2D surface.  The Hilbert curve visits every node exactly once,
threading through the 2D space.  Splitting the sequence at the midpoint gives
two groups that are **geometrically interleaved** across the whole surface:

```
2D fault viewed from above — letters show which half each node belongs to:

  A B A B A B A B
  B A B A B A B A      ← A and B alternate throughout the fault
  A B A B A B A B
  B A B A B A B A
```

Every node in cluster A has near-field neighbours in cluster B and vice versa.
The off-diagonal block A × B contains **near-field interactions** — these are
NOT low-rank.  The rank fraction climbs as blocks get smaller:

| HODLR level | Block size (rows) | Rank used | Rank / size |
|---|---|---|---|
| 1 | ~5520 | 275 | 5% |
| 2 | ~2760 | 163 | 6% |
| 3 | ~1380 | 180 | 13% |
| 4 | ~690  | 115 | 17% |
| 5 | ~344  |  96 | **28%** |
| 6 | ~172  |  86 | **50% — saturated** |
| 7 | ~86   | dense | 100% |

At level 6 the off-diagonal block requires rank ≈ half the block size.  This
cannot be reduced by increasing `max_rank` or `leaf_size`: the block is
intrinsically full-rank because the two clusters share a large near-field
boundary.

### Why the coarse levels still work

At level 1 the two clusters are the "left sweep" and "right sweep" of the entire
Hilbert curve.  Even though they are interleaved at fine scales, the dominant
part of the interaction is between nodes that are physically distant (the curve
visits both sides of the fault in each half).  The far-field contributions are
smooth and low-rank.  As the hierarchy deepens, the far-field fraction shrinks
and the near-field fraction grows until it dominates completely at levels 5–6.

---

## The Fix: 2D Spatial (k-d tree) ClusterTree

Instead of partitioning by MPI rank (1D sequence), partition by the **actual 2D
spatial coordinates** of the fault nodes using a k-d tree.

### How it works

```
build_spatial_tree(node_indices, coords_2d, leaf_size):

    if |node_indices| ≤ leaf_size:
        return leaf node

    // k-d split: cut perpendicular to the longest axis
    dim  = argmax over d of (max_i coords[i][d] - min_i coords[i][d])
    med  = spatial median of coords[·][dim]
    left  = { i : coords[i][dim] ≤ med }
    right = { i : coords[i][dim] > med }

    return internal_node(
        build_spatial_tree(left,  coords_2d, leaf_size),
        build_spatial_tree(right, coords_2d, leaf_size)
    )
```

With this tree every off-diagonal block at level l corresponds to two
**spatially separated rectangles** on the fault.  Off-diagonal interactions
between them are purely far-field and the rank is bounded by a small constant
times `O(log(N))` per level (for the elastic GF in 3D).

### Expected outcome for bp7

| Level | Clusters (kd-tree) | Expected rank |
|---|---|---|
| 1 | top half vs bottom half of fault | ~20–50 |
| 2 | four quadrants of fault | ~20–50 |
| 3 | eight sub-rectangles | ~15–30 |
| ... | ... | ... |
| leaf | single spatial tile (dense) | stored as dense |

The near-field blocks (spatially adjacent tiles) are absorbed into the **dense
leaf storage** by construction — they are leaves, not off-diagonal blocks.

### Implementation notes

1. `build_spatial_tree` uses the projected 2D fault coordinates already computed
   by the PCA step (`eff_dim = 2`, `proj[]` array in `build_spatial_permutation`).

2. The spatial tree does **not** need to respect MPI rank boundaries.  The
   allgatherv bridge in the MatMult callback already handles arbitrary
   distributions.

3. The Hilbert ordering (used for the within-leaf ordering) is kept: it
   determines the ordering of nodes *within* each leaf block and is still
   beneficial for data locality.

4. For the `leaf_size` parameter: with a spatial tree, this directly controls
   the minimum tile size.  Setting `leaf_size = 256–512` for bp7 (Np=5520)
   gives ~3 levels of hierarchy, each compressible, with dense tiles at the
   bottom.

### MPI tree alignment (optional optimisation)

For very large problems, performance can be improved by aligning the spatial
tree's leaf boundaries with PETSc's MPI rank boundaries.  This avoids allgather
in the callback (zero-copy VecPlaceArray instead).  For current problem sizes
(Np ≲ 1e5) the allgatherv cost is negligible and alignment is not required.

---

## Summary

| | bp3 (2D, 1D fault) | bp7 (3D, 2D fault) |
|---|---|---|
| Hilbert ordering | eff\_dim=1, line | eff\_dim=2, curve on surface |
| HODLR partition | left / right — **separated** | interleaved — **near-field mixed in** |
| Fine-level rank | 6–7 (< 1% of block) | 86–96 (50–100% of block) |
| Current result | PASS, ~4000x compression | FAIL, error ≈ 5 |
| Fix needed | — | 2D spatial ClusterTree (k-d tree) |

The HODLR compression failure on 3D faults is not a tuning issue (`leaf_size`,
`max_rank`, `rtol`).  It is a structural mismatch between the 1D Hilbert-curve
partition and the 2D geometry of the fault.  The solution is to replace
`build_petsc_tree` with a coordinate-based k-d tree that guarantees spatial
separation at every level of the hierarchy.
