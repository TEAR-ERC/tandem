import pytest


@pytest.mark.parametrize("scenario", ["free_slip_extension", "free_slip_compression"])
@pytest.mark.parametrize("label", ["L2 error", "H1-semi error"])
def test_free_slip_bc(
    temp_results_path, tolerances, domain_dim, extract_static_errors, scenario, label
):
    """
    Check that the static solution of the free slip block matches the exact
    (linear) solution up to solver tolerance, see examples/elasticity/2d/free_slip.lua
    and its 3D analogue examples/elasticity/3d/free_slip.lua.
    """
    log_file = temp_results_path / f"{scenario}_{domain_dim}D.log"
    assert log_file.exists(), f"Missing log file: {log_file}"

    errors = extract_static_errors(log_file, label)
    assert errors, (
        f"No '{label}:' entries found in {log_file}. "
        "Check the log for 'Solver did not converge.'"
    )

    error = errors[-1]
    tol = tolerances["static"]
    assert error < tol, f"{label} too large for {scenario}: {error:.2e} >= {tol:.2e}"
