"""
Spatial and temporal convergence of the viscoelastic operator against a
time-dependent manufactured solution, run through app/tandem (QD) with the
lua 'force' hook, see examples/tandem/viscoelasticity/2d/mms.lua and
scripts/2D/viscoelastic_mms.sh.

Both fields are divergence free, so the volumetric branch drops out and the
deviatoric memory path is what is tested, and both start from rest, which the
code requires (eps^{dev,n} is carried from zero).

- Spatial: trigonometric field (scenario mms), theta = 0.01, three resolutions.
  The L2 error at t = 1 must converge at order p + 1.
- Temporal: polynomial field (scenario mms_poly), fixed coarse mesh, four values
  of theta. The field lies in the discrete space for p >= 2, so the L2 error at
  t = 1 is the time-integration error alone and must converge at order 2, with
  no spatial floor to stop the fit early.
"""

import numpy as np
import pytest


def compute_slope(x_vals, y_vals):
    """Slope of a log-log least-squares fit (order of convergence)."""
    slope, _ = np.polyfit(np.log(x_vals), np.log(y_vals), 1)
    return slope


def local_orders(x_vals, y_vals):
    """Order between each pair of neighbouring levels."""
    x = np.log(np.asarray(x_vals))
    y = np.log(np.asarray(y_vals))
    return np.diff(y) / np.diff(x)


def read_runs(temp_results_path, pattern, read_tandem_summary):
    runs = [read_tandem_summary(f) for f in sorted(temp_results_path.glob(pattern))]
    assert runs, f"No logs matching {pattern} in {temp_results_path}"
    for run in runs:
        assert "L2_error_domain" in run, (
            "L2_error_domain= missing from the summary; does the scenario define 'solution'?"
        )
    return runs


def check_time_steps(run, conf):
    """dt is capped at theta * tau and final_time is a multiple of it, so every step
    is exactly theta * tau: a broken cap shows up here first."""
    dt = run["max_time_step"]
    assert run["min_time_step"] == pytest.approx(dt, rel=conf["rel_dt"]), (
        f"variable time step: min {run['min_time_step']:.10e}, max {dt:.10e}"
    )
    assert run["time_steps"] == round(conf["final_time"] / dt), (
        f"time_steps={run['time_steps']:.0f}, expected final_time / dt = {conf['final_time'] / dt:.6f}"
    )


def test_spatial_convergence(
    temp_results_path, read_tandem_summary, viscoelastic_mms_config, polynomial_degree
):
    """L2 error of the trigonometric field at t = 1 converges at order p + 1."""
    conf = viscoelastic_mms_config
    space = conf["space"]
    runs = read_runs(temp_results_path, "viscoelastic_mms_space_*.log", read_tandem_summary)
    assert len(runs) == space["num_levels"], (
        f"expected {space['num_levels']} resolutions, found {len(runs)} logs"
    )

    for run in runs:
        check_time_steps(run, conf)
        theta = run["max_time_step"] / conf["tau"]
        assert theta == pytest.approx(space["theta"], rel=conf["rel_dt"]), (
            f"spatial study ran at theta = {theta}, expected {space['theta']}"
        )

    runs.sort(key=lambda r: r["mesh_size"])
    h = [r["mesh_size"] for r in runs]
    l2 = [r["L2_error_domain"] for r in runs]
    order = compute_slope(h, l2)

    expected, tol = space["expected_order"], space["order_tol"]
    assert abs(order - expected) < tol, (
        f"Spatial L2 order {order:.4f} at p{polynomial_degree} is outside "
        f"{expected} +- {tol} (h = {h}, L2 = {l2})"
    )


def test_temporal_convergence(
    temp_results_path, read_tandem_summary, viscoelastic_mms_config, polynomial_degree
):
    """
    L2 error of the polynomial field at t = 1 converges at order 2 in theta.
    Every local order is checked as well as the fit: if the finest theta ever
    reaches a floor (spatial or solver), the last local order collapses and the
    test says so instead of reporting a slightly low slope.
    """
    conf = viscoelastic_mms_config
    temporal = conf["time"]
    if polynomial_degree < temporal["min_degree"]:
        pytest.skip(f"the polynomial manufactured field needs p >= {temporal['min_degree']}")

    runs = read_runs(temp_results_path, "viscoelastic_mms_time_*.log", read_tandem_summary)
    assert len(runs) == temporal["num_levels"], (
        f"expected {temporal['num_levels']} values of theta, found {len(runs)} logs"
    )
    for run in runs:
        check_time_steps(run, conf)

    runs.sort(key=lambda r: r["max_time_step"])
    theta = [r["max_time_step"] / conf["tau"] for r in runs]
    l2 = [r["L2_error_domain"] for r in runs]
    assert len({r["mesh_size"] for r in runs}) == 1, "temporal study must use one mesh"

    expected, tol = temporal["expected_order"], temporal["order_tol"]
    order = compute_slope(theta, l2)
    assert abs(order - expected) < tol, (
        f"Temporal L2 order {order:.4f} is outside {expected} +- {tol} "
        f"(theta = {theta}, L2 = {l2})"
    )
    for (t_fine, t_coarse), local in zip(zip(theta[:-1], theta[1:]), local_orders(theta, l2)):
        assert abs(local - expected) < tol, (
            f"Local temporal order {local:.4f} between theta = {t_fine} and {t_coarse} "
            f"is outside {expected} +- {tol}: a floor is reached or the scheme lost "
            f"its order (theta = {theta}, L2 = {l2})"
        )
