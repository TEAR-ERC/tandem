import pytest


def _extract_errors(log_file, label):
    errors = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{label}:"):
                try:
                    errors.append(float(line.split(":", 1)[1].strip()))
                except ValueError:
                    pass
    return errors


@pytest.mark.parametrize("scenario", ["free_slip_extension", "free_slip_compression"])
@pytest.mark.parametrize("label", ["L2 error", "H1-semi error"])
def test_free_slip_bc(temp_results_path, tolerances, domain_dim, scenario, label):
    """
    Check that the static solution of the free slip block matches the exact
    (linear) solution up to solver tolerance, see reference_configs/2D/free_slip.lua.
    """
    log_file = temp_results_path / f"{scenario}_{domain_dim}D.log"
    assert log_file.exists(), f"Missing log file: {log_file}"

    errors = _extract_errors(log_file, label)
    assert errors, (
        f"No '{label}:' entries found in {log_file}. "
        "Check the log for 'Solver did not converge.'"
    )

    error = errors[-1]
    tol = tolerances["static"]
    assert error < tol, f"{label} too large for {scenario}: {error:.2e} >= {tol:.2e}"
