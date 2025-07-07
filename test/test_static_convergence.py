import math
import numpy as np


def parse_static_log(log_file):
    data = []
    current_entry = {}

    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("DOFs:"):
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
                        (
                            current_entry["h"],
                            current_entry["dofs"],
                            current_entry["residual"],
                            current_entry["l2_error"],
                            current_entry["h1_error"],
                        )
                    )
                    current_entry = {}

    data.sort(key=lambda x: x[0])  # sort by mesh size
    return data


def compute_slope(x_vals, y_vals):
    log_x = np.log(x_vals)
    log_y = np.log(y_vals)
    slope, _ = np.polyfit(log_x, log_y, 1)
    return slope


def compute_all_slopes(data):
    mesh_size = np.array([entry[0] for entry in data])
    l2_errors = np.array([entry[3] for entry in data])
    h1_errors = np.array([entry[4] for entry in data])

    slope_l2 = compute_slope(mesh_size, l2_errors)
    slope_h1 = compute_slope(mesh_size, h1_errors)

    return slope_l2, slope_h1


def test_convergence(request, tolerance_convergence=1e-12):
    domain_dimension = request.config.getoption("domain_dimension")
    log_file = f"test_data/temp_test_results/convergence_{domain_dimension}D.log"
    data = parse_static_log(log_file)
    l2_order, h1_order = compute_all_slopes(data)

    # Precomputed orders of convergence for 2 and 3 dimensions
    expected = {
        "2": (1.5105428398689058, 1.4294570164157487),
        "3": (2.6718624932533888, 2.606395508949031),
    }
    expected_l2, expected_h1 = expected[str(domain_dimension)]

    assert math.isclose(
        l2_order, expected_l2, rel_tol=tolerance_convergence
    ), f"Computed L2 order {l2_order} does not match expected {expected_l2}"
    assert math.isclose(
        h1_order, expected_h1, rel_tol=tolerance_convergence
    ), f"Computed H1 order {h1_order} does not match expected {expected_h1}"
