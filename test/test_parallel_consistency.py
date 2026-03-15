import glob


def compare_one_vs_many(
    reference_file,
    file_prefix,
    number_of_configs,
    field_name,
    load_vtu_file,
    get_cell_centroid_point_dofs,
    compute_l2_error_with_reference_data,
    tolerances,
):
    """
    Compare a reference VTU file to a set of partitioned VTU outputs for a particular field. Asserts that the L2 norm of the difference is below the provided tolerance for each partitioning configuration.
    """
    ref_data = load_vtu_file(reference_file)
    output_procs = [2**j for j in range(1, number_of_configs)]
    vtu_files = [f"{file_prefix}{i}_0.vtu" for i in output_procs]

    ref_dict = get_cell_centroid_point_dofs(ref_data, field_name)
    for idx, vtu_file in enumerate(vtu_files):
        data = load_vtu_file(vtu_file)
        cmp_dict = get_cell_centroid_point_dofs(data, field_name)
        L2_norm = compute_l2_error_with_reference_data(ref_dict, cmp_dict)
        assert (
            L2_norm < tolerances["static"]
        ), f"L2 error too large for {output_procs[idx]} procs."


def _run_parallel_consistency(
    domain_dimension,
    output_prefix_name,
    temp_results_path,
    load_vtu_file,
    get_cell_centroid_point_dofs,
    compute_l2_error_with_reference_data,
    tolerances,
):
    reference_file = (
        temp_results_path / f"{output_prefix_name}{domain_dimension}D_1_0.vtu"
    )
    output_prefix = temp_results_path / f"{output_prefix_name}{domain_dimension}D_"
    files = glob.glob(f"{output_prefix}*.vtu")
    number_of_configs = len(files)
    assert number_of_configs > 1

    ref_data = load_vtu_file(reference_file)
    point_data = ref_data.GetPointData()
    field_names = [
        point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())
    ]
    assert len(field_names) > 0, "No fields found in reference file"

    for field_name in field_names:
        compare_one_vs_many(
            reference_file,
            output_prefix,
            number_of_configs,
            field_name,
            load_vtu_file,
            get_cell_centroid_point_dofs,
            compute_l2_error_with_reference_data,
            tolerances,
        )


def test_parallel_consistency(
    request,
    temp_results_path,
    load_vtu_file,
    get_cell_centroid_point_dofs,
    compute_l2_error_with_reference_data,
    tolerances,
):
    """Test parallel consistency for standard static output."""
    _run_parallel_consistency(
        request.config.getoption("domain_dimension"),
        "parallel_output",
        temp_results_path,
        load_vtu_file,
        get_cell_centroid_point_dofs,
        compute_l2_error_with_reference_data,
        tolerances,
    )


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
