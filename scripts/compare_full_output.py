import numpy as np
import pyvista as pv
import argparse
import os
from tqdm import tqdm


def load_mesh(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    mesh = pv.read(filename)
    return mesh


def compute_pointwise_differences(
    mesh1: pv.UnstructuredGrid, mesh2: pv.UnstructuredGrid
):
    """Compute pointwise differences for common fields between two meshes,
    matched by cell centroids."""

    common_fields = set(mesh1.point_data.keys()) & set(mesh2.point_data.keys())
    for to_remove in ["partition", "vtkValidPointMask", "lambda", "mu"]:
        common_fields.discard(to_remove)
    if not common_fields:
        raise ValueError("No common fields found in point data")

    # Build cell-centroid → {point_coord: points_id} maps
    def get_centroid_map(mesh):
        result = {}
        cells = mesh.cell_connectivity.copy()
        n_points = len(mesh.get_cell(0).point_ids)
        cells = cells.reshape((mesh.n_cells, n_points))

        for cell_id in tqdm(range(mesh.n_cells)):
            centroid = mesh.get_cell(cell_id).center
            result[centroid] = cells[cell_id]
        return result

    centroids1 = get_centroid_map(mesh1)
    centroids2 = get_centroid_map(mesh2)

    assert set(centroids1) == set(centroids2), "Centroid sets differ"

    field_errors = {}

    for field in tqdm(common_fields):
        data1 = mesh1.point_data[field]
        data2 = mesh2.point_data[field]
        errs = np.zeros_like(data1)
        rel_errs = np.zeros_like(data1)
        for i, centroid in enumerate(centroids1):
            ids1 = centroids1[centroid]
            ids2 = centroids2[centroid]
            errs[i] = np.average(np.abs((data1[ids1] - data2[ids2])))
            ref = np.average(np.abs((data1[ids1])))
            if ref > 0:
                rel_errs[i] = errs[i] / ref

        field_errors[field] = (errs, rel_errs)

    return field_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare all common fields between two VTU/PVTU "
            "files by spatial matching."
        )
    )
    parser.add_argument("file1", help="Reference file (.vtu or .pvtu)")
    parser.add_argument("file2", help="Comparison file (.vtu or .pvtu)")
    args = parser.parse_args()

    mesh1 = load_mesh(args.file1)
    mesh2 = load_mesh(args.file2)

    field_errors = compute_pointwise_differences(mesh1, mesh2)

    for field, errors in field_errors.items():
        abs_err, rel_err = errors
        print(
            f"{field}: L2 = {np.sqrt(np.mean(abs_err**2)):.3e}, "
            f"L∞ = {np.max(abs_err):.3e}, "
            f"L1 = {np.mean(abs_err):.3e}, "
            f"relative errors: L2 = {np.sqrt(np.mean(rel_err**2)):.3e}, "
            f"L∞ = {np.max(rel_err):.3e}, "
            f"L1 = {np.mean(rel_err):.3e}"
        )
