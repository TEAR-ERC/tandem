import vtk
import argparse
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy


def read_dataset(filename):
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def create_probe_grid(bounds, spacing):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    nx = int((x_max - x_min) / spacing) + 1
    ny = int((y_max - y_min) / spacing) + 1
    nz = int((z_max - z_min) / spacing) + 1

    image = vtk.vtkImageData()
    image.SetOrigin(x_min, y_min, z_min)
    image.SetSpacing(spacing, spacing, spacing)
    image.SetDimensions(nx, ny, nz)
    npoints = nx * ny * nz
    print(f"probe_grid_dims: {nx} {ny} {nz} ({npoints} points)")
    return image


def probe(source, target):
    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetSourceData(target)
    probe_filter.SetInputData(source)
    probe_filter.Update()
    return probe_filter.GetOutput()


def get_array_name(point_data):
    narrays1 = point_data.GetNumberOfArrays()
    return {point_data.GetArrayName(i) for i in range(narrays1)}


def compute_and_save_differences(
    sampled1, sampled2, output_filename="field_differences.vti"
):
    point_data1 = sampled1.GetPointData()
    point_data2 = sampled2.GetPointData()

    arrays1 = get_array_name(point_data1)
    arrays2 = get_array_name(point_data2)
    common = arrays1 & arrays2

    for to_remove in ["partition", "vtkValidPointMask", "lambda", "mu"]:
        common.discard(to_remove)

    diff_grid = vtk.vtkImageData()
    diff_grid.DeepCopy(sampled1)  # Copy geometry and metadata

    for name in sorted(common):
        arr1 = point_data1.GetArray(name)
        arr2 = point_data2.GetArray(name)

        n = arr1.GetNumberOfTuples()
        comp = arr1.GetNumberOfComponents()

        diff_array = vtk.vtkDoubleArray()
        diff_array.SetName(f"{name}_diff")
        diff_array.SetNumberOfComponents(comp)
        diff_array.SetNumberOfTuples(n)

        max_diff = 0.0
        max_rel_diff = 0.0
        for i in range(n):
            for j in range(comp):
                v1 = arr1.GetComponent(i, j)
                v2 = arr2.GetComponent(i, j)
                diff = v1 - v2
                diff_array.SetComponent(i, j, diff)

        # Convert VTK array to NumPy array
        abs_err = np.abs(vtk_to_numpy(diff_array)).reshape((n, comp))
        ref = np.abs(vtk_to_numpy(arr1)).reshape((n, comp))
        rel_err = np.zeros_like(abs_err)
        mask = ref != 0
        rel_err[mask] = abs_err[mask] / ref[mask]

        print(
            f"{name}: L2 = {np.sqrt(np.mean(abs_err**2)):.3e}, "
            f"L∞ = {np.max(abs_err):.3e}, "
            f"L1 = {np.mean(abs_err):.3e}, "
            f"relative errors: L2 = {np.sqrt(np.mean(rel_err**2)):.3e}, "
            f"L∞ = {np.max(rel_err):.3e}, "
            f"L1 = {np.mean(rel_err):.3e}"
        )

        diff_grid.GetPointData().AddArray(diff_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(diff_grid)
    writer.Write()

    print(f"Saved differences to {output_filename}")


def compare_by_probing(f1, f2, spacing):
    print(f"Reading: {f1}")
    mesh1 = read_dataset(f1)
    print(f"Reading: {f2}")
    mesh2 = read_dataset(f2)

    bounds = mesh1.GetBounds()
    print(f"Using bounds: {bounds}")
    probe_grid = create_probe_grid(bounds, spacing)

    print("Probing mesh1...")
    sampled1 = probe(probe_grid, mesh1)
    print("Probing mesh2...")
    sampled2 = probe(probe_grid, mesh2)

    print("Comparing fields...")
    compute_and_save_differences(
        sampled1, sampled2, output_filename="field_differences.vti"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two PVTU datasets by probing on a regular grid."
    )
    parser.add_argument("file1", help="First input file (.vtu or .pvtu)")
    parser.add_argument("file2", help="Second input file (.vtu or .pvtu)")
    parser.add_argument(
        "--spacing",
        type=float,
        default=5.0,
        help="Grid spacing for probing (default: 5.0)",
    )
    args = parser.parse_args()

    compare_by_probing(args.file1, args.file2, spacing=args.spacing)
