# Plan: `gf-hmatrix-visualize` — H-Matrix Structure Visualization App

## Goal

Build a new standalone app that assembles the H-matrix for a given TOML config (identical to
`gf-hmatrix-validate`) and exports its block-leaf structure to a CSV file, then visualize it
with an adapted version of the HTool Python plotting script.

---

## Key Design Decisions

### 1. Accessing the HTool `HMatrix` from a PETSc `Mat`

`MatHtoolGetHierarchicalMat` (the public API) returns a `const htool::DistributedOperator<PetscScalar>*`.
This is **not** the `HMatrix` — it's the distributed operator wrapper.
The actual `htool::HMatrix<PetscScalar>` lives inside PETSc's private `Mat_Htool` struct:

```cpp
// petsc-3.22.3/src/mat/impls/htool/htool.hpp
struct Mat_Htool {
    ...
    std::unique_ptr<htool::DistributedOperatorFromHMatrix<PetscScalar>> distributed_operator_holder;
};
// distributed_operator_holder->hmatrix  is the HMatrix (public member)
```

**Approach**: include PETSc's private header and cast `H_->data`:

```cpp
#include <petsc-src-path/src/mat/impls/htool/htool.hpp>  // gives Mat_Htool
const Mat_Htool* impl = static_cast<const Mat_Htool*>(H_->data);
const auto& hmat = impl->distributed_operator_holder->hmatrix;
```

This is an internal API but stable across our locked PETSc version (3.22.3). We pin it to the
same PETSc source tree we already build against, so no external risk.

The PETSc source path is already known at build time via CMake's `PETSC_DIR`.

### 2. MPI output strategy

In an MPI run each rank owns a block-row of the global M×N matrix. `save_leaves_with_rank`
from HTool writes LOCAL offsets (relative to that rank's block start). For visualization we need
GLOBAL offsets.

Strategy: write a **custom per-rank export** that uses `get_offset()` (global cluster offset)
directly rather than subtracting the local-block start. Then rank 0 gathers all leaf records
and writes a single `<prefix>.csv` that covers the whole matrix.

```
CSV format (HTool compatible):
  row 0:  M, N           (global dims)
  row k:  row_start, row_count, col_start, col_count, compression_rank
           (-1 means dense block)
```

Per-rank files `<prefix>_rank<r>.csv` are also written for debugging.

### 3. What we export

| Output | Description |
|--------|-------------|
| `<prefix>.csv` | Complete leaf structure (global offsets, all ranks merged) |
| `<prefix>_rank<r>.csv` | Per-rank leaf structure (local offsets, HTool native format) |
| stdout | `print_distributed_hmatrix_information` + `print_tree_parameters` |

---

## Files to Create / Modify

| File | Action | Purpose |
|------|--------|---------|
| `app/gf-hmatrix-visualize.cpp` | **Create** | App entry-point (mirrors `gf-hmatrix-validate.cpp`) |
| `app/form/SeasQDDiscreteGreenOperator.h` | **Edit** | Add `export_h_structure(prefix)` declaration |
| `app/form/SeasQDDiscreteGreenOperator.cpp` | **Edit** | Implement leaf export + info print |
| `app/tandem/SEAS.h` | **Edit** | Add `dumpGFHMatrixStructure()` free function |
| `app/tandem/SEAS.cpp` | **Edit** | Implement `dumpGFHMatrixStructure()` |
| `app/CMakeLists.txt` | **Edit** | Add `gf-hmatrix-visualize` executable |
| `scripts/plot_hmatrix.py` | **Create** | Python visualization script |

---

## Step-by-Step Implementation

### Step 1 — `SeasQDDiscreteGreenOperator`: add `export_h_structure()`

In `SeasQDDiscreteGreenOperator.h`, add inside the `#ifdef PETSC_HAVE_HTOOL` block:

```cpp
// Export H-matrix leaf structure for visualization.
// Writes <prefix>.csv (global, merged) and <prefix>_rank<r>.csv (per-rank).
// Also prints HTool information to stdout on rank 0.
void export_h_structure(const std::string& prefix) const;
```

In `SeasQDDiscreteGreenOperator.cpp`:

```cpp
#ifdef PETSC_HAVE_HTOOL
// Include PETSc's private HTool header to access Mat_Htool::distributed_operator_holder.
// This is pinned to PETSc 3.22.3 in our build — update if PETSc is upgraded.
#include <petsc/src/mat/impls/htool/htool.hpp>  // Mat_Htool, DistributedOperatorFromHMatrix

void SeasQDDiscreteGreenOperator::export_h_structure(const std::string& prefix) const {
    if (!H_) return;

    MPI_Comm comm = base::comm();
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    // --- Access the underlying HTool HMatrix via PETSc internal struct ---
    const Mat_Htool* impl = static_cast<const Mat_Htool*>(H_->data);
    const auto& hmat = impl->distributed_operator_holder->hmatrix;

    // --- Print structural info to stdout ---
    if (rank == 0) {
        htool::print_tree_parameters(hmat, std::cout);
        htool::print_distributed_hmatrix_information(hmat, std::cout, comm);
    }

    // --- Write per-rank CSV (local offsets, HTool native format) ---
    htool::save_leaves_with_rank(hmat, prefix + "_rank" + std::to_string(rank));

    // --- Gather leaves and write global CSV on rank 0 ---
    // Collect local leaves with global offsets
    struct Leaf { int row0, nrows, col0, ncols, crank; };
    std::vector<Leaf> local_leaves;

    using HM = htool::HMatrix<PetscScalar, PetscReal>;
    htool::preorder_tree_traversal(hmat,
        [&local_leaves](const HM& node) {
            if (node.is_leaf()) {
                local_leaves.push_back({
                    static_cast<int>(node.get_target_cluster().get_offset()),  // global row start
                    static_cast<int>(node.get_target_cluster().get_size()),
                    static_cast<int>(node.get_source_cluster().get_offset()),  // global col start
                    static_cast<int>(node.get_source_cluster().get_size()),
                    node.get_rank()   // compression rank; -1 = dense
                });
            }
        });

    // Gather leaf counts, then leaf data to rank 0
    int local_count = static_cast<int>(local_leaves.size());
    std::vector<int> counts(nranks);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);

    // Pack leaves as flat int array (5 ints per leaf)
    constexpr int FIELDS = 5;
    std::vector<int> flat(local_count * FIELDS);
    for (int i = 0; i < local_count; ++i) {
        flat[i*FIELDS+0] = local_leaves[i].row0;
        flat[i*FIELDS+1] = local_leaves[i].nrows;
        flat[i*FIELDS+2] = local_leaves[i].col0;
        flat[i*FIELDS+3] = local_leaves[i].ncols;
        flat[i*FIELDS+4] = local_leaves[i].crank;
    }

    std::vector<int> displs(nranks, 0);
    std::vector<int> recv_counts(nranks);
    for (int r = 0; r < nranks; ++r) recv_counts[r] = counts[r] * FIELDS;
    for (int r = 1; r < nranks; ++r) displs[r] = displs[r-1] + recv_counts[r-1];
    int total = displs[nranks-1] + recv_counts[nranks-1];

    std::vector<int> all_leaves;
    if (rank == 0) all_leaves.resize(total);
    MPI_Gatherv(flat.data(), local_count * FIELDS, MPI_INT,
                all_leaves.data(), recv_counts.data(), displs.data(), MPI_INT, 0, comm);

    if (rank == 0) {
        PetscInt M, N;
        MatGetSize(H_, &M, &N);
        std::ofstream out(prefix + ".csv");
        out << M << "," << N << "\n";
        int n_leaves = total / FIELDS;
        for (int i = 0; i < n_leaves; ++i) {
            out << all_leaves[i*FIELDS+0] << ","   // row_start
                << all_leaves[i*FIELDS+1] << ","   // nrows
                << all_leaves[i*FIELDS+2] << ","   // col_start
                << all_leaves[i*FIELDS+3] << ","   // ncols
                << all_leaves[i*FIELDS+4] << "\n"; // compression rank
        }
        std::cout << "Wrote " << n_leaves << " leaves to " << prefix << ".csv\n";
    }
}
#endif
```

### Step 2 — `SEAS.h` / `SEAS.cpp`: add `dumpGFHMatrixStructure()`

In `SEAS.h`:
```cpp
void dumpGFHMatrixStructure(LocalSimplexMesh<DomainDimension> const& mesh,
                            Config const& cfg,
                            const std::string& output_prefix);
```

In `SEAS.cpp`, follow the same pattern as `validateGFHMatrix`:
- set `cfg_h.hmatrix_config.use_hmatrix = true`
- build `seasop` via `detail::make_context` + `detail::operator_specifics::make`
- call `seasop->export_h_structure(output_prefix)`
- under `#ifdef PETSC_HAVE_HTOOL` guard

### Step 3 — `gf-hmatrix-visualize.cpp`

Clone `gf-hmatrix-validate.cpp` and:
- rename the `argparse::ArgumentParser` program name
- add `--output` argument for the output file prefix (default: `hmatrix_structure`)
- call `dumpGFHMatrixStructure(*mesh, *cfg, output_prefix)` instead of `validateGFHMatrix`
- no return-code logic based on error (this app always succeeds if HTool is present)

### Step 4 — `CMakeLists.txt`

Add after the `gf-hmatrix-validate` block:

```cmake
add_executable(gf-hmatrix-visualize
    tandem/Monitor.cpp
    tandem/SeasConfig.cpp
    tandem/SEAS.cpp
    gf-hmatrix-visualize.cpp)
target_link_libraries(gf-hmatrix-visualize PRIVATE app-common)
target_include_directories(gf-hmatrix-visualize PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PETSC_DIR}/src)   # needed for petsc/src/mat/impls/htool/htool.hpp
```

The extra include path `${PETSC_DIR}/src` exposes the private `htool.hpp`. This is compile-time
only and does not affect the installed library or ABI.

### Step 5 — `scripts/plot_hmatrix.py`

Adapt the HTool reference script with the following changes:

- Support merged multi-rank CSV (global offsets, single file): **default mode**
- Color scheme:
  - Red = dense block (rank == -1)
  - Green gradient = compressed block, shade proportional to compression rank
  - Black rectangle outlines on all blocks
  - Rank annotations inside large blocks (> 5% of matrix width)
- Optional `--ranks N` argument: if given, draw vertical dashed lines at the row partition
  boundaries (to see which block-row each MPI rank owns)
- Output: `--save <file>` saves as PNG/PDF; `--show` opens interactive window

```python
# Usage:
#   python scripts/plot_hmatrix.py --inputfile hmatrix_structure.csv [--save out.png] [--ranks 4]
```

The script is a direct adaptation of the HTool reference — main additions are the global-offset
support (already handled because we write global offsets in the merged CSV) and the MPI partition
lines.

---

## CMake `PETSC_DIR` Handling

`PETSC_DIR` is already used in tandem's CMake to find PETSc. To expose the source path to the
private header, add to `CMakeLists.txt`:

```cmake
# Only needed by gf-hmatrix-visualize (private PETSc header for HTool HMatrix access)
if(DEFINED ENV{PETSC_DIR})
    set(PETSC_SRC_DIR $ENV{PETSC_DIR})
else()
    # Fall back: PETSC_DIR from the find_package result
    get_filename_component(PETSC_SRC_DIR "${PETSC_DIR}" DIRECTORY)
endif()
```

Then use `${PETSC_SRC_DIR}/src` in the target include path above.

---

## Usage

```bash
# Build
cmake --build build --target gf-hmatrix-visualize

# Run (same TOML as tandem/validate, mode=QDGreen, [hmatrix] enabled)
mpirun -n 4 ./build/app/gf-hmatrix-visualize bp1.toml --output hmat_bp1

# Visualize
python scripts/plot_hmatrix.py --inputfile hmat_bp1.csv --ranks 4 --save hmat_bp1.png
```

---

## Caveats / Open Questions

| Item | Note |
|------|------|
| `htool.hpp` private header | Pinned to PETSc 3.22.3. Must re-check if PETSc is upgraded. |
| `preorder_tree_traversal` | Declared in `htool/basic_types/tree.hpp` — already transitively included via `htool.hpp`. |
| Large matrices | For N > 10k the resulting PNG can be very dense; consider `--dpi 300` or writing SVG instead. |
| Serial run | Works fine; only one rank, one CSV file, no gather needed. |
| `save_leaves_with_rank` format | The native HTool per-rank files use LOCAL offsets (relative to the rank's block). Our merged `.csv` uses GLOBAL offsets — the Python script reads both but they are NOT interchangeable. |
