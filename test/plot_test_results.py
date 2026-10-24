"""
Plots of the static convergence, SEAS regression, boundary condition and
viscoelastic test results, for the CI artifacts. Not a test: run as its own ctest entry (plot-test-results, see
CMakeLists.txt) after generate-test-outputs and before the cleanup, with
--plot_dir pointing to where the PNGs go.

It only reads the logs and probe files the tests already read; it runs no
simulation. The closed forms and limits come from the test modules and the
conftest fixtures, so a plot always shows what the test checks.

A plot whose inputs are missing (other dimension, test not built) or that fails
is skipped with the reason, so this never fails the CI; the tests themselves
report the failures.
"""

from contextlib import contextmanager

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import test_convergence_static as convergence  # noqa: E402
import test_regression_SEAS as seas  # noqa: E402
import test_viscoelastic_cantilever as cantilever  # noqa: E402
import test_viscoelastic_mms as mms  # noqa: E402
import test_viscoelastic_simple_shear as shear  # noqa: E402

DPI = 100
YEAR = 365.25 * 24 * 3600


@pytest.fixture(scope="module")
def save(plot_dir, domain_dim, polynomial_degree):
    """Save a figure as <plot_dir>/<name>_<dim>D_p<degree>.png and close it."""

    def _save(fig, name):
        path = plot_dir / f"{name}_{domain_dim}D_p{polynomial_degree}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")

    return _save


@contextmanager
def plotting(name):
    """Turn a missing input or a plotting error into a skip with the reason."""
    try:
        yield
    except FileNotFoundError as e:
        plt.close("all")
        pytest.skip(f"{name}: input missing ({e})")
    except Exception as e:
        plt.close("all")
        pytest.skip(f"{name}: plotting failed ({type(e).__name__}: {e})")


def test_plot_static_bc(temp_results_path, tolerances, domain_dim, extract_static_errors, save):
    """L2 and H1-semi errors of the free slip and traction runs against the static tolerance."""
    runs = [f"free_slip_{s}" for s in ("extension", "compression")]
    if domain_dim == 2:
        runs += [
            f"traction_patch_{s}{r}"
            for s in ("tension", "compression")
            for r in ("", "_flipped_normal")
        ]
    with plotting("static_bc"):
        labels, l2, h1 = [], [], []
        for run in runs:
            log = temp_results_path / f"{run}_{domain_dim}D.log"
            if not log.exists():
                raise FileNotFoundError(log)
            e_l2 = extract_static_errors(log, "L2 error")
            e_h1 = extract_static_errors(log, "H1-semi error")
            labels.append(run.replace("_flipped_normal", "\n(ref_normal -x)"))
            # A run without an error line did not converge: draw it at the top.
            l2.append(e_l2[-1] if e_l2 else np.nan)
            h1.append(e_h1[-1] if e_h1 else np.nan)

        tol = tolerances["static"]
        x = np.arange(len(runs))
        fig, ax = plt.subplots(figsize=(1.0 + 1.1 * len(runs), 3.5))
        floor = 1e-17
        ax.bar(x - 0.2, np.maximum(np.nan_to_num(l2, nan=1.0), floor), 0.4, label="L2 error")
        ax.bar(x + 0.2, np.maximum(np.nan_to_num(h1, nan=1.0), floor), 0.4, label="H1-semi error")
        ax.axhline(tol, color="k", ls="--", lw=1, label=f"limit {tol:.0e}")
        ax.set_yscale("log")
        ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("error against the exact solution")
        ax.set_title(f"Free slip and traction BC tests, {domain_dim}D (bar at 1: no error in log)")
        ax.legend(fontsize=8)
        save(fig, "static_bc_errors")


def test_plot_static_convergence(
    temp_results_path, convergence_config, polynomial_degree, domain_dim, save
):
    """L2 and H1-semi errors against h, log-log, with the band the fitted slope must lie in."""
    with plotting("static_convergence"):
        log = temp_results_path / f"convergence_{domain_dim}D.log"
        if not log.exists():
            raise FileNotFoundError(log)
        data = convergence.parse_static_log(log)
        if not data:
            raise FileNotFoundError(f"{log} has no complete entries")
        h = np.array([d.h for d in data])

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        for ax, (norm, label) in zip(axes, (("l2", "L2 error"), ("h1", "H1-semi error"))):
            err = np.array([getattr(d, f"{norm}_error") for d in data])
            lower, upper = convergence_config[norm]["lower"], convergence_config[norm]["upper"]
            slope = convergence.compute_slope(h, err)
            ax.loglog(h, err, "o-", label=f"tandem, slope {slope:.3f}")
            # Band edges through the finest point: the fitted slope must lie strictly between.
            for order, style in ((lower, ":"), (upper, "--")):
                ax.loglog(h, err[0] * (h / h[0]) ** order, "k", ls=style, lw=1,
                          label=f"order {order:g}")
            ax.set_xticks(h, [f"{v:.3g}" for v in h])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("h")
            ax.set_ylabel(label)
            ax.set_title(f"{label}, limit ({lower:g}, {upper:g})", fontsize=10)
            ax.legend(fontsize=8)
        name = "circular hole" if domain_dim == 2 else "spherical hole"
        fig.suptitle(f"Static convergence, {name}, {domain_dim}D p{polynomial_degree}")
        save(fig, "static_convergence")


def test_plot_seas_regression(
    temp_results_path, reference_results_path, seas_config, polynomial_degree, domain_dim, save
):
    """VMax of this run over the stored reference, QD and QDGreen, with the detected events."""
    if domain_dim != 2:
        pytest.skip("the SEAS regression test is a 2D test")
    with plotting("seas_regression"):
        win, rel = seas_config["window_size"], seas_config["relative_error"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for ax, mode in zip(axes, ("QD", "QDGreen")):
            files = {
                "reference": reference_results_path / f"vmax_ref_{mode}.csv",
                "this run": temp_results_path / f"vmax_output_{mode}.csv",
            }
            for f in files.values():
                # detect_events returns [] for a missing file; a skip says more.
                if not f.exists():
                    raise FileNotFoundError(f)
            for (label, f), style, color in zip(files.items(), ("-", "--"), ("k", "C1")):
                df = pd.read_csv(f, comment="#")
                events = np.array(seas.detect_events(f, win, rel)).reshape(-1, 2)
                ax.semilogy(df["Time"] / YEAR, df["VMax"], style, color=color, lw=1,
                            label=f"{label} ({len(events)} events)")
                ax.semilogy(events[:, 0] / YEAR, events[:, 1], "v", color=color, ms=6)
            ax.set_xlabel("t [years]")
            ax.set_title(f"{mode}, p{polynomial_degree} (triangles: detected events)", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
        axes[0].set_ylabel("VMax [m/s]")
        fig.suptitle("SEAS regression, bp1_ref")
        save(fig, "seas_regression_vmax")


def test_plot_simple_shear(
    temp_results_path, read_domain_probe, viscoelastic_simple_shear_config, domain_dim, save
):
    """stress_xy against the closed form for the six loadings, and the error."""
    if domain_dim != 2:
        pytest.skip("simple shear is a 2D test")
    with plotting("simple_shear"):
        conf = viscoelastic_simple_shear_config
        model = shear.SimpleShearSLS(conf)

        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        fig_e, ax_e = plt.subplots(figsize=(7, 3.5))
        for ax, case in zip(axes.flat, shear.CASES):
            t, sig = shear.load_case(case, temp_results_path, read_domain_probe)
            ana = model.stress(case, t)
            rel_mae, _ = shear.relative_errors(sig, ana)
            if case in shear.EXACT_CASES:
                limit = conf["exact"]
            else:
                limit = conf["headroom"] * conf["measured"][case]["rel_mae"]

            tt = np.linspace(0.0, t[-1], 2000)
            ax.plot(tt / model.tau, model.stress(case, tt), "k-", lw=1, label="closed form")
            ax.plot(t / model.tau, sig, "o", ms=2, color="C0", label="tandem")
            ax.set_title(case.replace("_", " "), fontsize=9)
            ax.text(0.98, 0.95, f"rel. MAE {rel_mae:.2e}\nlimit {limit:.2e}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7)
            # Normalised as in the test (mean |sigma|), so cases of any stress scale compare.
            ax_e.semilogy(t / model.tau, np.maximum(np.abs(sig - ana) / np.abs(ana).mean(), 1e-17),
                          lw=1, label=case.replace("_", " "))
        for ax in axes[1]:
            ax.set_xlabel("t / tau")
        for ax in axes[:, 0]:
            ax.set_ylabel("stress_xy")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Viscoelastic simple shear (SLS), center probe")
        save(fig, "viscoelastic_simple_shear_stress")

        ax_e.set_xlabel("t / tau")
        ax_e.set_ylabel("|stress_xy - closed form| / mean|stress_xy|")
        ax_e.set_title("Viscoelastic simple shear, pointwise error")
        ax_e.legend(fontsize=7)
        save(fig_e, "viscoelastic_simple_shear_error")


def test_plot_mms(
    temp_results_path, read_tandem_summary, viscoelastic_mms_config, polynomial_degree,
    domain_dim, save,
):
    """L2 error against h (spatial study) and against theta (temporal study), log-log."""
    if domain_dim != 2:
        pytest.skip("the manufactured solution is a 2D test")
    with plotting("mms"):
        conf = viscoelastic_mms_config
        studies = [("space", "viscoelastic_mms_space_*.log", "mesh_size", "h")]
        if polynomial_degree >= conf["time"]["min_degree"]:
            studies.append(("time", "viscoelastic_mms_time_*.log", "max_time_step", "theta"))

        fig, axes = plt.subplots(1, len(studies), figsize=(4.5 * len(studies), 3.8), squeeze=False)
        for ax, (study, pattern, key, xlabel) in zip(axes[0], studies):
            runs = sorted((read_tandem_summary(f) for f in temp_results_path.glob(pattern)),
                          key=lambda r: r[key])
            if not runs:
                raise FileNotFoundError(temp_results_path / pattern)
            x = np.array([r[key] for r in runs])
            if study == "time":
                x = x / conf["tau"]
            err = np.array([r["L2_error_domain"] for r in runs])
            expected = conf[study]["expected_order"]

            ax.loglog(x, err, "o-", label=f"tandem, slope {mms.compute_slope(x, err):.3f}")
            ax.loglog(x, err[0] * (x / x[0]) ** expected, "k--", lw=1,
                      label=f"order {expected:g} (limit +-{conf[study]['order_tol']:g})")
            ax.set_xticks(x, [f"{v:.3g}" for v in x])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel(xlabel)
            ax.set_ylabel("L2 error at t = 1")
            ax.set_title(f"{study} convergence, p{polynomial_degree}", fontsize=10)
            ax.legend(fontsize=8)
        fig.suptitle("Viscoelastic manufactured solution")
        save(fig, "viscoelastic_mms_convergence")


def test_plot_cantilever(
    temp_results_path, read_domain_probe, viscoelastic_cantilever_config, domain_dim, save
):
    """eps_xx(t) against the Zener closed form, its error, and the stress output error."""
    if domain_dim != 3:
        pytest.skip("the rod creep test is a 3D test")
    with plotting("cantilever"):
        conf = viscoelastic_cantilever_config
        rod = cantilever.ZenerRod(conf)
        probes = {
            name: read_domain_probe(
                temp_results_path / cantilever.WORK_DIR / f"cantilever_rod_probe_{name}.csv"
            )
            for name in cantilever.PROBES
        }
        data, _ = probes["free_end"]
        t = data["Time"].to_numpy()
        eps = cantilever.strain_history(data, rod)
        exact = rod.strain(t)

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        tt = np.linspace(0.0, t[-1], 2000)
        axes[0].plot(tt, rod.strain(tt), "k-", lw=1, label="Zener closed form")
        axes[0].plot(t, eps, "o", ms=2, color="C0", label="tandem, free end")
        axes[0].set_ylabel("eps_xx")
        axes[0].set_title("creep and recovery", fontsize=10)

        axes[1].semilogy(t, np.maximum(np.abs(eps - exact), 1e-17), lw=1, label="tandem, free end")
        creep = conf["headroom"] * conf["measured"]["creep_abs"]
        rebound = conf["headroom"] * rod.rebound_excess(conf["theta"])
        axes[1].hlines(creep, 0.0, rod.t1, color="k", ls="--", lw=1,
                       label=f"creep limit {creep:.1e}")
        axes[1].hlines(rebound, rod.t1, t[-1], color="k", ls=":", lw=1,
                       label=f"rebound limit {rebound:.1e}")
        axes[1].set_ylim(bottom=1e-9)
        axes[1].set_ylabel("|eps_xx - closed form|")
        axes[1].set_title("strain error (rebound overshoot at t1 is expected)", fontsize=10)

        loaded = t <= rod.t1
        for name in cantilever.PROBES:
            d, _ = probes[name]
            rel = np.abs(d["stress_xx"].to_numpy() / rod.s0 - 1.0)
            axes[2].semilogy(t[loaded], np.maximum(rel[loaded], 1e-17), lw=1, label=name)
        axes[2].axhline(conf["stress_rel"], color="k", ls="--", lw=1,
                        label=f"limit {conf['stress_rel']:.0e}")
        axes[2].set_ylabel("|stress_xx / s0 - 1|")
        axes[2].set_title("stress output while loaded", fontsize=10)

        for ax in axes:
            ax.axvline(rod.t1, color="0.5", lw=0.8, ls=":")
            ax.set_xlabel("t [s]")
            ax.legend(fontsize=7)
        fig.suptitle("Viscoelastic rod creep and recovery (3D)")
        save(fig, "viscoelastic_cantilever")
