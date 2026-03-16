import glob
from test_parallel_consistency import _run_parallel_consistency


def test_parallel_consistency_volume_tagging(
    request,
    temp_results_path,
    load_vtu_file,
    get_cell_centroid_point_dofs,
    compute_l2_error_with_reference_data,
    tolerances,
):
    """Test parallel consistency for volume-tagged output."""
    _run_parallel_consistency(
        request.config.getoption("domain_dimension"),
        "parallel_volume_output",
        temp_results_path,
        load_vtu_file,
        get_cell_centroid_point_dofs,
        compute_l2_error_with_reference_data,
        tolerances,
    )
