import os
import numpy as np
import vtk
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def tolerance():
    return 1e-8


@pytest.fixture(scope="module")
def domain_dim(request):
    return request.config.getoption("domain_dimension")


def pytest_addoption(parser):
    parser.addoption(
        "--domain_dimension", action="store", help="Domain dimension (2 or 3)"
    )


@pytest.fixture(scope="module")
def reference_results_path():
    return Path("test_data/reference_results")


@pytest.fixture(scope="module")
def temp_results_path():
    return Path("test_data/temp_test_results")


@pytest.fixture(scope="module")
def load_vtu_file():
    def _load(file_path):
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return None

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(file_path)
        reader.Update()
        return reader.GetOutput()

    return _load


@pytest.fixture(scope="module")
def get_cell_centroid_point_dofs():
    def _get(data, field_name):
        point_data = data.GetPointData()
        field_array = point_data.GetArray(field_name)

        if field_array is None:
            raise ValueError(f"Field '{field_name}' not found in point data")

        field_components = field_array.GetNumberOfComponents()
        n_cells = data.GetNumberOfCells()

        point_to_value_mapping = {}
        for cell_id in range(n_cells):
            cell = data.GetCell(cell_id)
            point_ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
            coords = [data.GetPoint(pid) for pid in point_ids]
            dimensions = 3
            centroid = tuple(
                sum(p[i] for p in coords) / len(coords) for i in range(dimensions)
            )

            if field_components == 1:
                point_to_value_mapping[centroid] = {
                    tuple(data.GetPoint(pid)): field_array.GetValue(pid)
                    for pid in point_ids
                }
            else:
                point_to_value_mapping[centroid] = {
                    tuple(data.GetPoint(pid)): [
                        field_array.GetComponent(pid, j)
                        for j in range(field_components)
                    ]
                    for pid in point_ids
                }
        return point_to_value_mapping

    return _get


@pytest.fixture(scope="module")
def compute_l2_error_with_reference_data():
    def _compute(ref_dict, cmp_dict):
        coords = []
        errors = []
        for centroid, points in ref_dict.items():
            if centroid not in cmp_dict:
                print(f"Centroid {centroid} not found in comparison data")
                continue

            cmp_points = cmp_dict[centroid]
            for point, ref_dof in points.items():
                if point not in cmp_points:
                    print(
                        f"Point {point} not found in comparison data for centroid {centroid}"
                    )
                    continue

                cmp_dof = cmp_points[point]
                abs_diff = abs(cmp_dof - ref_dof)
                coords.append(point)
                errors.append(abs_diff)
        L2_norm = np.sqrt(np.sum(np.square(errors)))
        return L2_norm

    return _compute


@pytest.fixture(scope="module")
def compare_one_vs_many_processor_outputs(
    load_vtu_file,
    get_cell_centroid_point_dofs,
    compute_l2_error_with_reference_data,
    tolerance,
):
    def _compare(reference_file, file_prefix, number_of_configurations, field_name):
        ref_data = load_vtu_file(reference_file)
        output_procs = [2**j for j in range(1, number_of_configurations)]
        vtu_files = [f"{file_prefix}{i}_0.vtu" for i in output_procs]

        ref_dict = get_cell_centroid_point_dofs(ref_data, field_name)
        for idx, vtu_file in enumerate(vtu_files):
            data = load_vtu_file(vtu_file)
            cmp_dict = get_cell_centroid_point_dofs(data, field_name)
            L2_norm = compute_l2_error_with_reference_data(ref_dict, cmp_dict)
            assert (
                L2_norm < tolerance
            ), f"Output for {output_procs[idx]} procs is inconsistent with output for 1 proc."

    return _compare
