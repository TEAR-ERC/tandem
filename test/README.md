# Tandem test summary

This README provides a summary of the testing framework available in tandem. For detailed implementation information, please refer to the individual test scripts.

## Build System and Test Execution

The testing framework is built using a single `CMakeLists.txt` file located in the main test folder, which generates the Makefile target for testing. To run tests, use:

```bash
make test
```

### Test Configuration

- **Unit Tests**: Run locally by default
- **Integration Tests**: Require compilation with `-DRUN_CI_TESTS=ON`
- **Test Data**: All required test data is available in a separate repository, imported as a submodule : [tandem_test_data](https://github.com/piyushkarki/tandem_test_data/tree/master) (Note: Submodule updates are tied to commit hashes.)

## Test Categories

The test suite is organized into two main categories:

### 1. C++ Unit Tests

Fine-grained tests that verify specific functionalities of the codebase.

### 2. Pytest Integration Tests

Comprehensive tests that validate the overall functionality of tandem.

## C++ Unit Tests

| Test File       | Description                                                                     |
| --------------- | ------------------------------------------------------------------------------- |
| `basis.cpp`     | Validates correct evaluation of basis functions                                 |
| `form.cpp`      | Verifies transformation and mass matrices for specific degrees                  |
| `geometry.cpp`  | Tests building blocks of the Curvilinear class and geometry construction        |
| `io.cpp`        | Validates mesh parsing and importing functionality                              |
| `parallel.cpp`  | Verifies correct distribution patterns across multiple ranks                    |
| `quadrules.cpp` | Tests correctness of implemented quadrature rules                               |
| `script.cpp`    | Validates Lua script loading functionality                                      |
| `simplex.cpp`   | Tests simplex creation and upward/downward dimensional mapping                  |
| `tensor.cpp`    | Validates tensor creation and value copying operations                          |
| `util.cpp`      | Tests utility functions including combinatorics, root finding, and permutations |

## Pytest Integration Tests

### Reference Data Generation

All reference data is generated using:

- **PETSc version**: 3.22.x
- **gmsh version**: 4.12.1

Reference data is generated using the `generate_test_outputs.sh` script.

### Available Tests

| Test File                      | Description                                                                                                                                                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_regression_static.py`    | **Static Regression Testing**<br>Runs circular hole (2D) and spherical hole (3D) examples, comparing results against saved reference data                                                                        |
| `test_regression_SEAS.py`      | **SEAS Regression Testing**<br>Executes bp1 sym problem and validates:<br>- Event occurrence intervals<br>- Peak occurrence timing<br>- Slip rate magnitude during events                                        |
| `test_static_convergence.py`   | **Convergence Testing**<br>Runs circular hole (2D) and spherical hole (3D) with varying mesh sizes, verifying that the regression line slope matches precomputed values (L2 vs mesh size log-log plot)           |
| `test_parallel_consistency.py` | **Parallel Consistency Testing**<br>Executes circular hole (2D) and spherical hole (3D) with 2^n processors (n ∈ [0,3]), comparing single-processor vs multi-processor solutions to ensure parallelism integrity |

## Contributing

Update this README to reflect new additions to the test suite.