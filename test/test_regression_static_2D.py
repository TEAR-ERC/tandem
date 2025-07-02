import meshio
from numpy.testing import assert_allclose
import os


def test_static_solver_vtu_consistency_2D(results_path, tolerance_static):
    ref_path = results_path / "ref_regression_2D_output.vtu"
    test_path = results_path / "output2D_0.vtu"

    assert os.path.exists(test_path), f"Output file {test_path} not found."
    assert os.path.exists(ref_path), f"Reference file {ref_path} not found."

    ref = meshio.read(ref_path)
    out = meshio.read(test_path)

    field = "u0"
    assert field in ref.point_data, f"Field '{field}' not found in reference data"
    assert field in out.point_data, f"Field '{field}' not found in output data"

    ref_data = ref.point_data[field]
    out_data = out.point_data[field]

    assert_allclose(out_data, ref_data, atol=tolerance_static)
