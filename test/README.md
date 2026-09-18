# Tandem Test Suite Summary

This document provides a comprehensive guide to the testing framework available in Tandem. The test suite is divided into C++ Unit Tests for fine-grained component checks and Pytest Integration Tests for overall simulation validation.

## Test Directory Structure

Test configurations and generator scripts are maintained directly within this repository to keep tests closely coupled with the code.

- **`reference_configs/`**: Contains simulation configurations (`.toml`, `.lua`, `.geo`) separated by dimension (`2D/`, `3D/`).

- **`scripts/`**: Contains the individual shell scripts used to generate reference output data for each test.

- **`test_data/`**: The destination folder where the external `tandem_test_data` repository is cloned (only when integration tests are enabled).

## Build System and Execution

The testing framework is built using a single `CMakeLists.txt` file located in the main test folder.

### Running Basic Unit Tests

By default, compiling the tests will only prepare the local C++ unit tests. Run them using:

```bash
make test
```

### Running Integration Tests (CI)

Integration tests require heavy reference data (meshes, Green's functions, `.vtu` files). To enable them, compile with:

```bash
cmake -DRUN_CI_TESTS=ON ..
```

> **Note:** This flag automatically clones the `tandem_test_data` repository into the local `test_data/` directory.

### Testing with Custom Data Branches

By default, the CI setup pulls the `master` branch of the `tandem_test_data` repository. If you are developing a new feature and need a custom branch of the test data, you can specify it during configuration using `-DTEST_DATA_BRANCH`.

**Example Workflow:**
If you are working on a Tandem branch named `<user-name>/add-feature-1`, you can create a corresponding data branch `<user-name>/add-data-for-feature-1` in the data repo and compile Tandem with:

```bash
cmake -DRUN_CI_TESTS=ON -DTEST_DATA_BRANCH=<user-name>/add-data-for-feature-1 ..
```

## Test Categories

### 1. C++ Unit Tests

Fine-grained tests verifying specific codebase functionalities.

| Test File | Description |
|---|---|
| `basis.cpp` | Validates correct evaluation of basis functions |
| `form.cpp` | Verifies transformation and mass matrices for specific degrees |
| `geometry.cpp` | Tests building blocks of the Curvilinear class and geometry construction |
| `io.cpp` | Validates mesh parsing and importing functionality |
| `parallel.cpp` | Verifies correct distribution patterns across multiple ranks |
| `quadrules.cpp` | Tests correctness of implemented quadrature rules |
| `script.cpp` | Validates Lua script loading functionality |
| `simplex.cpp` | Tests simplex creation and upward/downward dimensional mapping |
| `tensor.cpp` | Validates tensor creation and value copying operations |
| `util.cpp` | Tests utility functions (combinatorics, root finding, permutations) |
| `HDF5MeshParser.cpp` | Basic parsing, Boundary tag decoding, Face deduplication, node validation, error handling, compilation fallback |
| `HDF5Writer.cpp`| File creation, dataset extension, timestep writes, checkpointing (append/overwrite), data integrity, file closure, read-back validation, compilation fallback |



### 2. Pytest Integration Tests

Comprehensive tests validating the physics, parallelism, and convergence of the Tandem solver.

- **PETSc version:** 3.22.x
- **gmsh version:** 2.2
- Each integration test relies on its own specific generator script located in the `scripts/2D` or `scripts/3D` directories.

| Test File | Description |
|---|---|
| `test_regression_static.py` | **Static Regression:** Runs circular (2D) and spherical (3D) hole examples, comparing results against saved reference data. |
| `test_regression_SEAS.py` | **SEAS Regression:** Executes bp1 sym problem. Validates event occurrence intervals, peak timing, and slip rate magnitudes. |
| `test_convergence_static.py` | **Convergence:** Runs hole examples with varying mesh sizes. Verifies the regression line slope (L2 vs mesh size log-log plot) matches precomputed values. |
| `test_parallel_consistency_static.py` | **Parallel Consistency:** Executes static examples using 2^n processors (n ∈ [0,3]). Compares single vs multi-processor solutions to ensure MPI integrity. |
| `test_probe_writer_consistency.py` | **HDF5 probe writer consistency:** Runs two bp1 simulations with the same probe locations but different probe writer formats (csv and h5) and compares that the two results are identical for moment rate, fault probes and domain probes. |

## Adding New Tests

We encourage adding tests for new features! Follow these steps based on the type of test:

### Adding a Unit Test

1. Create a new `.cpp` script in the root test directory.
2. Add your new test file to the `CMakeLists.txt` target list.

### Adding an Integration Test

1. **Add Configurations:** Place your simulation configuration files (`.toml`, `.lua`, `.geo`) into the appropriate dimensional folder (`reference_configs/2D/` or `reference_configs/3D/`).

2. **Write a Generator Script:** Create a shell script in the corresponding `scripts/2D/` or `scripts/3D/` folder to generate the output data.
   - **Crucial:** Ensure your script automatically cleans up any temporary test output files after execution to prevent local clutter.

3. **Write the Pytest:** Create a `test_<feature>.py` script in the root test folder to execute your test logic and parameter checks.

4. **Update the External Data Repo (If needed):** If your test requires comparing against previously calculated data (e.g., `.vtu`, `.csv`) or needs pre-generated heavy inputs (e.g., meshes, Green's functions), push those files to the `tandem_test_data` repository. *(Use the branching strategy detailed above to test your changes before merging).*

> **IMPORTANT NOTICE FOR CONTRIBUTORS**
> If you add new tests (Unit or Integration), you **MUST** document them by updating the tables and descriptions in this `README.md` file. Keep the test documentation in sync with the codebase.
