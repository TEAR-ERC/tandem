import os
import numpy as np
import vtk
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def tolerances():
    """
    Centralized numerical tolerances for pass/fail criteria.

    - 'static': Threshold for L2/H1 errors in steady-state problems.
    - 'seas': Tolerance for time-stepping consistency in SEAS cycles.
    - 'seas_events': Looser tolerance for peak slip rates (VMax) during events.
    - 'spatial_match': Maximum distance allowed when identifying the same
       physical point across different mesh partitions.
    """
    return {
        "static": 1e-8,
        "seas": {"p2": 1e-2, "p3": 1e-8},
        "seas_events": {"QD": 1e-2, "QDGreen": 1e-2, "both": 1e-2},
        "spatial_match": 1e-8,
    }


@pytest.fixture(scope="module")
def seas_config():
    """
    Parameters for the Sequence of Earthquake and Aseismic Slip (SEAS)
    event detection algorithm.

    - 'window_size': The temporal 'look-ahead' (in seconds) used to group
       high slip-rate points into a single seismic event.
    - 'relative_error': The required drop in slip-rate magnitude to
       distinguish the end of one event from the start of another.
    """
    return {
        "window_size": 1e9,
        "relative_error": 0.5,
    }


@pytest.fixture(scope="module")
def traction_bc_config():
    """
    Physical and geometric constraints for Traction Boundary Condition tests.

    - 'x_target': The specific coordinate (the end of a rod) where
       analytical solutions are verified.
    - 'x_tol': Spatial window around x_target to account for mesh discretization.
    - 'expected_u0': The theoretical displacement value at x_target.
    """
    return {
        "x_target": 0.0001,
        "x_tol": 1e-8,
        "expected_u0": 5e-9,
    }


@pytest.fixture(scope="module")
def convergence_config(polynomial_degree):
    """
    Theoretical convergence rates for DG verification.
    For a polynomial basis of order N, we expect:
    - L2 error to reduce at a rate of N+1 (e.g., N=3 -> 4th order).
    - H1 error to reduce at a rate of N (e.g., N=3 -> 3rd order).

    The 'lower' and 'upper' bounds provide a buffer for numerical noise.
    """
    n = polynomial_degree
    return {
        "l2": {"lower": n, "upper": n + 1},
        "h1": {"lower": n - 1, "upper": n},
    }


@pytest.fixture(scope="module")
def domain_dim(request):
    """
    Fixture that exposes the requested domain dimension CLI option.

    The option is provided via --domain_dimension and is used to choose
    reference/test file names that differ between 2D and 3D runs.
    """
    return request.config.getoption("domain_dimension")


@pytest.fixture(scope="module")
def polynomial_degree(request):
    return request.config.getoption("--polynomial_degree")


def pytest_addoption(parser):
    """
    Add custom pytest CLI options used by the tests.

    Adds --domain_dimension to select 2D or 3D test artifacts.
    Adds --polynomial_degree to specify the polynomial degree for tests.
    """
    parser.addoption(
        "--domain_dimension", action="store", type=int, help="Domain dimension (2 or 3)"
    )
    parser.addoption(
        "--polynomial_degree",
        action="store",
        type=int,
        help="Polynomial degree (integer)",
    )


@pytest.fixture(scope="module")
def reference_results_path(polynomial_degree):
    """Path to reference outputs for the active polynomial degree."""
    return Path("./test_data/reference_results") / f"p{polynomial_degree}"


@pytest.fixture(scope="module")
def temp_results_path():
    """Path to the directory containing outputs produced by the current test run."""
    return Path("./temp_test_results")


@pytest.fixture(scope="module")
def load_vtu_file():
    """
    Return a helper that loads a VTU (vtkXMLUnstructuredGrid) file and
    returns the VTK data object. The helper returns None and prints an error if
    the file does not exist.
    """

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
    """
    Return a helper that builds a mapping from cell centroids to their
    vertex point DOF values for a specified field name.

    The returned function accepts (data, field_name) and returns a nested
    dictionary mapping centroid -> {point_coord: value_or_vector}.
    """

    def _get(data, field_name):
        """
        Build a mapping from cell centroids to their basis point DOF values that correspond to the vertices.

        The centroid is computed as the average of the cell's point coordinates,
        and is used as a spatial key to match cells across different VTU files
        (e.g. single-processor vs multi-processor output).

        Returns a nested dict of the form:
        {
            (0.5, 0.5, 0.0): {       # cell centroid (x, y, z)
                (0.0, 0.0, 0.0): 101.3,                   (1.0, 0.0, 0.0): 101.5,
                (1.0, 1.0, 0.0): 102.1,
            },
            ...
        }
        For vector fields, values are lists of components instead of scalars:
        {
            (0.5, 0.5, 0.0): {
                (0.0, 0.0, 0.0): [101.3, 0.5],
                ...
            }
        }

        """
        point_data = data.GetPointData()
        field_array = point_data.GetArray(field_name)

        if field_array is None:
            raise ValueError(f"Field '{field_name}' not found in point data")

        field_components = field_array.GetNumberOfComponents()
        n_cells = data.GetNumberOfCells()

        point_to_value_mapping = {}
        # VTK always returns 3D coordinates even for 2D problems (z=0)
        dimensions = 3
        for cell_id in range(n_cells):
            cell = data.GetCell(cell_id)
            point_ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
            coords = [data.GetPoint(pid) for pid in point_ids]
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
    """
    Return a helper that computes the L2 norm of differences between
    reference and comparison centroid->point mappings.

    The returned function returns a scalar L2 norm. centroids are matched by nearest-neighbor search within spatial_tol.
    """

    def _compute(ref_dict, cmp_dict, spatial_tol=1e-10):
        """
        Compute the L2 norm of point-wise differences between ref_dict and
        cmp_dict. Centroids are matched within spatial_tol; unmatched centroids
        are reported and ignored in the L2 calculation.
        """
        # Build a list of centroids from cmp_dict for efficient nearest neighbor search
        cmp_centroids = np.array(list(cmp_dict.keys()))
        errors = []
        unmatched_centroids = 0

        for centroid, points in ref_dict.items():
            # Find nearest centroid in cmp_dict within spatial_tol
            centroid_arr = np.array(centroid)
            distances = np.linalg.norm(cmp_centroids - centroid_arr, axis=1)
            nearest_idx = np.argmin(distances)

            if distances[nearest_idx] > spatial_tol:
                unmatched_centroids += 1
                continue

            matched_centroid = tuple(cmp_centroids[nearest_idx])
            cmp_points = cmp_dict[matched_centroid]

            # Build point array for nearest-neighbour lookup, same as centroids
            cmp_point_coords = np.array(list(cmp_points.keys()))

            for point, ref_dof in points.items():
                point_arr = np.array(point)
                pt_distances = np.linalg.norm(cmp_point_coords - point_arr, axis=1)
                nearest_pt_idx = np.argmin(pt_distances)

                if pt_distances[nearest_pt_idx] > spatial_tol:
                    print(
                        f"Point {point} not found in comparison data "
                        f"for centroid {centroid}"
                    )
                    continue

                matched_point = tuple(cmp_point_coords[nearest_pt_idx])
                cmp_dof = cmp_points[matched_point]
                abs_diff = abs(cmp_dof - ref_dof)
                errors.append(abs_diff)

        assert unmatched_centroids == 0, (
            f"{unmatched_centroids} centroids could not be matched "
            f"within spatial_tol={spatial_tol}"
        )

        L2_norm = np.sqrt(np.sum(np.square(errors)))
        return L2_norm

    return _compute
