import math
import numpy as np
from collections import namedtuple


def parse_static_log(log_file):
    LogEntry = namedtuple("LogEntry", ["h", "dofs", "residual", "l2_error", "h1_error"])
    data = []
    current_entry = {}

    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("DOFs:"):
                current_entry = {}  # reset at the start of each new entry
                current_entry["dofs"] = int(line.split()[1])
            elif line.startswith("Mesh size:"):
                current_entry["h"] = float(line.split()[2])
            elif line.startswith("Residual norm:"):
                current_entry["residual"] = float(line.split()[2])
            elif line.startswith("L2 error:"):
                current_entry["l2_error"] = float(line.split()[2])
            elif line.startswith("H1-semi error:"):
                current_entry["h1_error"] = float(line.split()[2])

                if all(
                    key in current_entry
                    for key in ["dofs", "h", "residual", "l2_error", "h1_error"]
                ):
                    data.append(
                        LogEntry(
                            h=current_entry["h"],
                            dofs=current_entry["dofs"],
                            residual=current_entry["residual"],
                            l2_error=current_entry["l2_error"],
                            h1_error=current_entry["h1_error"],
                        )
                    )
                    current_entry = {}

    data.sort(key=lambda x: x.h)  # sort by mesh size
    return data


def compute_slope(x_vals, y_vals):
    """Compute slope of a log-log fit between x_vals and y_vals (order of convergence)."""
    log_x = np.log(x_vals)
    log_y = np.log(y_vals)
    slope, _ = np.polyfit(log_x, log_y, 1)
    return slope


def compute_all_slopes(data):
    """
    Compute L2 and H1 convergence slopes from parsed log data.

    Returns a tuple (slope_l2, slope_h1).
    """
    mesh_size = np.array([entry.h for entry in data])
    l2_errors = np.array([entry.l2_error for entry in data])
    h1_errors = np.array([entry.h1_error for entry in data])

    slope_l2 = compute_slope(mesh_size, l2_errors)
    slope_h1 = compute_slope(mesh_size, h1_errors)

    return slope_l2, slope_h1


def test_convergence(request):
    """
    Integration test that verifies the computed convergence orders against
    precomputed expected values for the given domain dimension.
    """
    domain_dimension = request.config.getoption("domain_dimension")
    log_file = f"test_data/temp_test_results/convergence_{domain_dimension}D.log"
    data = parse_static_log(log_file)
    l2_order, h1_order = compute_all_slopes(data)

    # Expected order of convergence for order N is N+1 - for N=3, we expect 4th order convergence;
    N = 3
    L2_upper_bound = N + 1
    L2_lower_bound = N

    H1_upper_bound = N
    H1_lower_bound = N - 1

    assert l2_order > L2_lower_bound and l2_order < L2_upper_bound, (
        f"Computed L2 order {l2_order:.4f} "
        f"(rounded: {l2_order}) does not match expected order between {L2_lower_bound} and {L2_upper_bound}"
    )
    assert h1_order > H1_lower_bound and h1_order < H1_upper_bound, (
        f"Computed H1 order {h1_order:.4f} "
        f"(rounded: {h1_order}) does not match expected order between {H1_lower_bound} and {H1_upper_bound}"
    )
