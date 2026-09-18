import re

import pytest


def _extract_l2_errors(log_file):
    errors = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("L2 error:"):
                try:
                    value = float(line.split(":", 1)[1].strip())
                    errors.append(value)
                except ValueError:
                    pass
    return errors


def test_correctness_volume_tagging(request, temp_results_path, tolerances, domain_dim):
    """Check that L2 error from the volume-tagging correctness log is below tolerance."""

    log_file = temp_results_path / f"volume_tagging_correctness_{domain_dim}D.log"
    assert log_file.exists(), f"Missing log file: {log_file}"

    l2_errors = _extract_l2_errors(log_file)
    assert l2_errors, f"No 'L2 error:' entries found in {log_file}"

    l2_error = l2_errors[-1]
    tol = tolerances.get("volume_tagging", tolerances["static"])
    assert l2_error < tol, (
        f"L2 error too large for volume tagging: {l2_error} >= {tol}. "
        f"Parsed {len(l2_errors)} L2 entries from {log_file}."
    )
