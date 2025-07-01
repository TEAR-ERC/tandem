import glob


def compare_one_vs_many(
    reference_file,
    file_prefix,
    number_of_configs,
    field_name,
    load_vtu_file,
    get_cell_centroid_point_dofs,
    get_l2_error_with_reference_data,
    tolerance,
):
    ref_data = load_vtu_file(reference_file)
    output_procs = [2**j for j in range(1, number_of_configs)]
    vtu_files = [f"{file_prefix}{i}_0.vtu" for i in output_procs]

    ref_dict = get_cell_centroid_point_dofs(ref_data, field_name)
    for idx, vtu_file in enumerate(vtu_files):
        data = load_vtu_file(vtu_file)
        cmp_dict = get_cell_centroid_point_dofs(data, field_name)
        L2_norm = get_l2_error_with_reference_data(ref_dict, cmp_dict)
        assert L2_norm < tolerance, f"L2 error too large for {output_procs[idx]} procs."


def test_parallel_consistency_2D(
    results_path,
    load_vtu_file,
    get_cell_centroid_point_dofs,
    get_l2_error_with_reference_data,
    tolerance,
):
    reference_file = results_path / "output2D_1_0.vtu"
    output_prefix = results_path / "output2D_"
    field_name = "u0"
    files = glob.glob(f"{output_prefix}*.vtu")
    number_of_configs = len(files)
    assert number_of_configs > 1
    compare_one_vs_many(
        reference_file,
        output_prefix,
        number_of_configs,
        field_name,
        load_vtu_file,
        get_cell_centroid_point_dofs,
        get_l2_error_with_reference_data,
        tolerance,
    )
