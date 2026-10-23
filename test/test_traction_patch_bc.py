import pytest


@pytest.mark.parametrize("ref_normal", ["", "_flipped_normal"], ids=["ref_normal+x", "ref_normal-x"])
@pytest.mark.parametrize("scenario", ["traction_patch_tension", "traction_patch_compression"])
@pytest.mark.parametrize("label", ["L2 error", "H1-semi error"])
def test_traction_patch_bc(
    temp_results_path, tolerances, extract_static_errors, scenario, label, ref_normal
):
    """
    Check prescribed traction (tag 7) and traction-free (tag 1) boundaries in 2D:
    the static solution of the uniaxially loaded block must match the exact
    (linear, uniform-stress) solution up to solver tolerance, see
    examples/elasticity/2d/traction_patch.lua.

    The exact solution has sigma.n = 0 on the top and bottom, so this is also
    the explicit check of the Natural (traction-free) condition: a wrong sign,
    a missing term or a spurious penalty on those facets lands in the error
    norm. The block is run in tension and compression, so a traction sign error
    cannot cancel, and with ref_normal = [1, 0] and [-1, 0]: on boundary facets
    the prescribed traction is the global traction vector and must not depend
    on ref_normal.
    """
    log_file = temp_results_path / f"{scenario}{ref_normal}_2D.log"
    assert log_file.exists(), f"Missing log file: {log_file}"

    errors = extract_static_errors(log_file, label)
    assert errors, (
        f"No '{label}:' entries found in {log_file}. "
        "Check the log for 'Solver did not converge.'"
    )

    error = errors[-1]
    tol = tolerances["static"]
    assert error < tol, f"{label} too large for {scenario}{ref_normal}: {error:.2e} >= {tol:.2e}"
