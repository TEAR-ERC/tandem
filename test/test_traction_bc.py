import numpy as np


def check_u0_at_x(
    vtu_file,
    field_name,
    load_vtu_file,
    get_cell_centroid_point_dofs,
    tolerances,
    config,
):
    """
    Load a VTU file and assert that the L2 norm of (field_name - expected_u0)
    over all points with x ≈ x_target is below the static tolerance.
    """
    data = load_vtu_file(vtu_file)
    assert data is not None, f"Could not load VTU file: {vtu_file}"

    centroid_point_dofs = get_cell_centroid_point_dofs(data, field_name)

    x_target = config["x_target"]
    x_tol = config["x_tol"]
    expected_u0 = config["expected_u0"]

    diffs = []
    for _centroid, point_map in centroid_point_dofs.items():
        for point_coord, value in point_map.items():
            if abs(point_coord[0] - x_target) <= x_tol:
                scalar = value[0] if isinstance(value, list) else value
                diffs.append(scalar - expected_u0)

    assert (
        diffs
    ), f"No points found with x ≈ {x_target} (tol={x_tol}) in field '{field_name}'."

    l2_norm = np.sqrt(np.sum(np.square(diffs)))
    assert (
        l2_norm < tolerances["static"]
    ), f"L2 error {l2_norm:.2e} exceeds tolerance {tolerances['static']:.2e}"


def test_traction_bc(
    temp_results_path,
    load_vtu_file,
    get_cell_centroid_point_dofs,
    tolerances,
    traction_bc_config,
):
    """Test that u0 at x matches the expected value using config from conftest."""
    vtu_file = temp_results_path / "cantilever_rod_0.vtu"

    check_u0_at_x(
        vtu_file,
        "u0",
        load_vtu_file,
        get_cell_centroid_point_dofs,
        tolerances,
        traction_bc_config,
    )
