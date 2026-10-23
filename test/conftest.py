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
def expected_moment_rate_sum_t0():
    """
    Expected sum of per-element moment rates at first output timestep for bp1_ref 2D
    reference mesh (bp1_ref.geo, degree=2).

    Recompute from verified reference run if mesh/physics/quadrature changes.
    """
    return 1.281524803579644e-06


@pytest.fixture(scope="module")
def viscoelastic_simple_shear_config():
    """
    Material, loading and pass/fail limits for the viscoelastic simple shear test
    (test_viscoelastic_simple_shear.py, examples/tandem/viscoelasticity/2d/simple_shear.*).

    - Material and loading: copied from simple_shear.lua, which the test runs
      unchanged. mu0, mu1 in GPa, eta in GPa.s (tau = eta/mu1 = 7.8894e8 s),
      gamma(t) = 2*Umax/H * s(t), drop at t_hold_tau*tau, omega = omega_tau/tau.
    - 'theta': dt/tau of simple_shear.lua. The time step is capped at theta*tau,
      and every step of this run is that long except the last; checked against
      max_time_step= to rel_dt.
    - 'measured': relative MAE and RMSE of stress_xy at the center probe against
      the closed form, each normalised by the mean / RMS of |sigma|. Measured on
      2026-10-20 at np = 1, 2D p2, tandem afb4c3bb, direct LU solve. They agree
      with the 2026-09 dt-convergence campaign at theta = 0.05 to 3-4 digits.
      The error is the time integration alone: the field is linear in y, so the
      14-triangle mesh is exact in space. The with-drop cases are first order
      (the drop is a strain jump), the sinusoid second order.
    - 'headroom': limit = headroom * measured. 1.25 passes today and fails on a
      25% loss of accuracy, e.g. any change that turns a first-order error into
      twice its size.
    - 'exact': limit on the relative MAE and RMSE of the two cases whose strain
      history is piecewise linear in time (constant strain, constant strain
      rate). The exponential integrator is exact for them; measured 1.5e-15 and
      1.4e-15. This holds only because the solve is direct: with GCR + multigrid
      at ksp_rtol 1e-6 the floor is 6.6e-6 and tracks rtol, so a Krylov solver
      here would need a limit of that size and would hide a real error.
    - 'sigma0_rel': limit on sigma_xy(0) / ((mu0 + mu1) gamma(0)) - 1. The t = 0
      solve uses the unrelaxed modulus mu0 + mu1 (SeasQDOperator::initial_condition);
      measured 8.9e-16. Loadings with gamma(0) = 0 must give sigma_xy(0) = 0 to
      the same limit relative to (mu0 + mu1) * 2*Umax/H.
    """
    return {
        "mu0": 30.0,
        "mu1": 30.0,
        "eta": 2.36682e10,
        "umax": 1.0e-3,
        "height": 1.0,
        "t_hold_tau": 6.0,
        "omega_tau": 2.0,
        "theta": 0.05,
        "rel_dt": 1e-12,
        "measured": {
            "constant_strain_with_drop": {"rel_mae": 3.0779e-03, "rel_rmse": 5.8125e-03},
            "constant_strain_rate_with_drop": {"rel_mae": 5.2963e-03, "rel_rmse": 9.4575e-03},
            "sinusoidal_without_drop": {"rel_mae": 2.1127e-04, "rel_rmse": 2.1221e-04},
            "sinusoidal_with_drop": {"rel_mae": 1.9180e-03, "rel_rmse": 2.8734e-03},
        },
        "headroom": 1.25,
        "exact": 1e-10,
        "sigma0_rel": 1e-12,
    }


@pytest.fixture(scope="module")
def viscoelastic_mms_config(polynomial_degree):
    """
    Expected convergence orders for the viscoelastic manufactured solution test
    (test_viscoelastic_mms.py, examples/tandem/viscoelasticity/2d/mms.*).

    - 'space': L2 error at t = 1 of the trigonometric field (scenario mms) at
      theta = 0.01, resolutions 0.25, 0.125, 0.0625. Expected order p + 1.
      Measured slopes 2.96 (p2) and 4.02 (p3) on 2026-10-20 (np = 1, direct LU),
      i.e. on either side of p + 1, so the static convergence_config band
      (p, p + 1) does not fit; 'order_tol' is a band centred on p + 1 instead.
      The coarsest level 0.5 is left out because it is pre-asymptotic (local
      orders 3.38 at p2, 2.22 at p3 from 0.5 to 0.25). A drop to order p
      (an inconsistent assembly) fails the band by 0.75.
      Measured L2 at p2: 0.0133013, 0.00170094, 0.000219658 (reproducing the
      2026-09-29 sweep in /export/dump/pkarki/mms_ve_time on kernelpanic).
    - 'time': L2 error at t = 1 of the polynomial field (scenario mms_poly) on
      resolution 0.25, theta = 0.4, 0.2, 0.1, 0.05. Expected order 2 (the
      recursion is second order for smooth strain histories). The field lies in
      the discrete space for p >= 2, so there is no spatial floor: measured
      6.17024e-05, 1.53701e-05, 3.83429e-06, 9.57459e-07 at p2 (local orders
      2.005, 2.003, 2.002) and the same to 0.4% at p3. 'order_tol' bounds both
      the fitted slope and every local order, so a floor appearing at the
      finest theta (e.g. after a change that leaves the field out of the space)
      fails on the last local order instead of quietly lowering the fit.
      The same study with the trigonometric field needs resolution 0.03125 at
      p2 and still has its finest point only 1.55x above the spatial floor.
    - 'min_degree': the polynomial field needs p >= 2.
    - 'rel_dt': max_time_step= must equal theta * tau, and time_steps= must be
      final_time / (theta * tau), both to this relative tolerance.
    """
    return {
        "tau": 0.5,
        "final_time": 1.0,
        "space": {
            "expected_order": polynomial_degree + 1,
            "order_tol": 0.25,
            "theta": 0.01,
            "num_levels": 3,
        },
        "time": {
            "expected_order": 2.0,
            "order_tol": 0.1,
            "num_levels": 4,
            "min_degree": 2,
        },
        "rel_dt": 1e-12,
    }


@pytest.fixture(scope="module")
def viscoelastic_cantilever_config():
    """
    Material, loading and pass/fail limits for the viscoelastic cantilever rod
    creep-recovery test (test_viscoelastic_cantilever.py,
    examples/tandem/viscoelasticity/3d/cantilever_rod.*, meshed with clamped = 0).

    - Material and loading: copied from cantilever_rod.lua, which the test runs
      unchanged. G0 in GPa, mu0 = (1 - r1) G0, mu1 = r1 G0, eta in GPa.s, so
      tau = eta / mu1 = 100 s; K = lam + (2/3) G0 = 10 GPa; traction s0 = 0.01 GPa
      on x = L for t <= t1 = 6000 s. The uniaxial response is a Zener solid with
      E0 = 9 K G0 / (3 K + G0), Einf = 9 K mu0 / (3 K + mu0) and creep time
      tau_c = tau G0 / mu0 = 1010 s (manuscript, Sec. 4.2).
    - 'theta': dt/tau of cantilever_rod.lua; every step is theta * tau = 10 s.
    - 'eps0_rel': limit on eps_xx(0) / (s0 / E0) - 1 at every probe. eps_xx is
      u_x / x with x the probe position printed in the CSV header, which has 6
      significant digits, so this cannot be tighter than ~1e-6; measured 2.9e-11
      at the probe whose position prints exactly and <= 4e-7 at the others. A t = 0
      solve with the step-sized instead of the unrelaxed moduli gives
      eps_xx(0) = 0.010450, i.e. +4.5%.
    - 'creep_abs': limit on max |eps_xx - exact| for t <= t1. Measured 2.4605e-6,
      largest at t = tau_c, as in the manuscript (2.5e-6). The run is exact in
      space, so this is the time integration of the creep alone.
    - 'rebound': the step that removes the load uses the effective modulus
      B(dt) = mu0 + mu1 g(dt) instead of mu0 + mu1, so the elastic rebound
      overshoots by s0 (1/E_eff - 1/E0) = 4.5064e-4, E_eff = 9 K B / (3 K + B)
      (manuscript, Sec. 4.2). Measured error at the first sample after t1:
      -4.507446e-4, 1.00023 times that prediction. Limit: 'headroom' times the
      prediction. This is a known first-order error of the scheme at a load jump,
      not a bug; the limit catches it doubling, not it being there.
    - 'recovery_rms': limit on the RMS of eps_xx - exact for t > t1. Measured
      1.5957e-4, dominated by the decaying rebound error.
    - 'headroom': limit = headroom * measured (or predicted), as for simple shear.
    - 'stress_rel': limit on |stress_xx / s0 - 1| and on every other stress
      component / s0 while loaded. The exact state is uniform uniaxial stress.
      The output without the (2/3) mu1 tr(eps) I term of the Maxwell spring
      (tandem afb4c3bb before the fix) gives 6.7e-3.
    - 'rel_dt': max_time_step= must equal theta * tau to this relative tolerance.
    """
    return {
        "g0": 0.337086226656778804,
        "r1": 0.901,
        "eta": 30.3714690218,
        "bulk_modulus": 10.0,
        "traction": 0.01,
        "t_load": 6000.0,
        "final_time": 10000.0,
        "theta": 0.1,
        "eps0_rel": 1e-4,
        "measured": {
            "creep_abs": 2.4605e-06,
            "recovery_rms": 1.5957e-04,
        },
        "headroom": 1.25,
        "stress_rel": 1e-8,
        "rel_dt": 1e-12,
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
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"VTU file not found: {file_path}")

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(file_path))
        reader.Update()
        output = reader.GetOutput()

        if output is None or output.GetNumberOfPoints() == 0:
            raise RuntimeError(f"VTU file loaded but contains no data: {file_path}")

        return output

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


@pytest.fixture(scope="module")
def extract_static_errors():
    """
    Return a helper that collects every value printed after '<label>:' in a log
    of app/static, e.g. extract_static_errors(log, "L2 error") -> [1.3e-12].

    app/static prints 'L2 error:' and 'H1-semi error:' when the scenario defines
    'solution' and 'solution_jacobian'. An empty list usually means the solve
    failed; the log then contains 'Solver did not converge.'
    """

    def _extract(log_file, label):
        errors = []

        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{label}:"):
                    try:
                        errors.append(float(line.split(":", 1)[1].strip()))
                    except ValueError:
                        pass
        return errors

    return _extract


@pytest.fixture(scope="module")
def read_tandem_summary():
    """
    Return a helper that parses the '========= Summary =========' block that
    app/tandem prints at the end of a run into a dict, e.g.
    {'time_steps': 200.0, 'max_time_step': 0.005, 'L2_error_domain': 0.0133, ...}.

    Numeric entries are floats, the rest (date_time, code_version) strings.
    L2_error_domain= is only printed when the scenario defines 'solution'.
    """

    def _read(log_file):
        log_file = Path(log_file)
        if not log_file.exists():
            raise FileNotFoundError(f"tandem log not found: {log_file}")

        summary = {}
        in_summary = False
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "========= Summary =========":
                    in_summary = True
                    summary = {}
                elif in_summary and line.startswith("====="):
                    in_summary = False
                elif in_summary and "=" in line:
                    key, value = line.split("=", 1)
                    try:
                        summary[key] = float(value)
                    except ValueError:
                        summary[key] = value

        if not summary:
            raise RuntimeError(f"No summary block in {log_file}; did the run finish?")
        return summary

    return _read


@pytest.fixture(scope="module")
def read_domain_probe():
    """
    Return a helper that reads a domain probe CSV written by tandem and returns
    (data, x): a pandas DataFrame with the columns of the quoted header
    ("Time", "u0", ..., "stress_xx", ...) and the probe position x as a list.

    The position comes from the '# Station <name> (x = [...])' line. It is
    where the probe was actually evaluated, which may differ from the position
    requested in the .toml (the nearest point of a per-element sampling
    lattice), and it is printed with 6 significant digits.
    """
    import re

    import pandas as pd

    def _read(csv_file):
        csv_file = Path(csv_file)
        if not csv_file.exists():
            raise FileNotFoundError(f"Probe file not found: {csv_file}")

        with open(csv_file, "r", encoding="utf-8") as f:
            header = f.readline()
        match = re.search(r"\(x = \[([^\]]*)\]\)", header)
        if match is None:
            raise ValueError(f"No station position in the first line of {csv_file}")
        x = [float(v) for v in match.group(1).split(",")]

        data = pd.read_csv(csv_file, comment="#")
        if data.empty:
            raise ValueError(f"Probe file contains no samples: {csv_file}")
        return data, x

    return _read
