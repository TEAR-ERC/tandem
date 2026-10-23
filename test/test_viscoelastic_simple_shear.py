"""
Simple shear of a standard linear solid (SLS) under six imposed strain
histories, compared against the closed-form solution.

Runs examples/tandem/viscoelasticity/2d/simple_shear.{geo,lua,toml} through
app/tandem (QD, no fault), one --scenario per loading, see
scripts/2D/viscoelastic_simple_shear.sh. The box is under Dirichlet control on
all four sides with u_x linear in y, so the stress is uniform and the 14-cell
mesh is exact in space: what is measured is the time integration of the
Maxwell branch and the t = 0 solve, nothing else.

Closed form (SLS in shear, q the Maxwell-branch internal strain):

    sigma(t) = mu0 * gamma(t) + mu1 * q(t),
    q(t)     = gamma(0) exp(-t/tau) + int_0^t exp(-(t-s)/tau) gamma'(s) ds,

with gamma = 2*Umax/H * s(t) and tau = eta / mu1. A strain jump d at time T
adds d * exp(-(t-T)/tau) to q for t >= T, which is how every "with drop" branch
is obtained. Ported from simple_shear_suite/analyze_simple_shear.py, where all
six closed forms were checked against a direct numerical integration of
q' = -q/tau + gamma' (agreement to 1e-13 relative for the smooth cases, 1e-6
for the discontinuous ones, which is the integrator smearing the jump).

The drop is sampled with t <= T_hold, as in simple_shear.lua. Using < here
would compare a loaded sample against the post-drop branch and inject an error
of the full stress scale.
"""

import numpy as np
import pytest

CASES = [
    "constant_strain_without_drop",
    "constant_strain_with_drop",
    "constant_strain_rate_without_drop",
    "constant_strain_rate_with_drop",
    "sinusoidal_without_drop",
    "sinusoidal_with_drop",
]

# Strain history piecewise linear in time: the exponential integrator is exact.
EXACT_CASES = ["constant_strain_without_drop", "constant_strain_rate_without_drop"]


class SimpleShearSLS:
    """Closed-form strain and stress histories for the six loadings."""

    def __init__(self, config):
        self.mu0 = config["mu0"]
        self.mu1 = config["mu1"]
        self.tau = config["eta"] / config["mu1"]
        self.beta = 1.0 / self.tau
        # Engineering shear strain amplitude: u_x spans +-Umax over height H.
        self.E0 = 2.0 * config["umax"] / config["height"]
        self.omega = config["omega_tau"] / self.tau
        self.T_hold = config["t_hold_tau"] * self.tau

    def strain(self, case, t):
        t = np.asarray(t, float)
        if case.startswith("constant_strain_rate"):
            e = self.E0 * t
        elif case.startswith("constant_strain"):
            e = self.E0 * np.ones_like(t)
        elif case.startswith("sinusoidal"):
            e = self.E0 * np.sin(self.omega * t)
        else:
            raise ValueError(case)
        if case.endswith("with_drop"):
            e = np.where(t <= self.T_hold, e, 0.0)
        return e

    def _q_loaded(self, case, t):
        """Internal variable q(t) before any drop."""
        t = np.asarray(t, float)
        if case.startswith("constant_strain_rate"):
            # eps = E0 t, eps(0) = 0
            return self.E0 * self.tau * (1.0 - np.exp(-t / self.tau))
        if case.startswith("constant_strain"):
            # eps = E0 (step at t=0), q(0) = E0, no further forcing
            return self.E0 * np.exp(-t / self.tau)
        if case.startswith("sinusoidal"):
            # eps = A sin(wt), eps(0) = 0
            A, w, b = self.E0, self.omega, self.beta
            return (A * w / (b * b + w * w)) * (
                b * np.cos(w * t) + w * np.sin(w * t) - b * np.exp(-b * t)
            )
        raise ValueError(case)

    def stress(self, case, t):
        t = np.asarray(t, float)
        sig = self.mu0 * self.strain(case, t) + self.mu1 * self._q_loaded(case, t)

        if case.endswith("with_drop"):
            Th = self.T_hold
            # q just before the drop, plus the jump -eps(Th^-) it induces
            q_h = float(self._q_loaded(case, np.array([Th]))[0])
            base = case.replace("_with_drop", "_without_drop")
            eps_h = float(self.strain(base, np.array([Th]))[0])
            q_after = q_h - eps_h
            post = self.mu1 * q_after * np.exp(-(t - Th) / self.tau)
            sig = np.where(t <= Th, sig, post)
        return sig


def load_case(case, temp_results_path, read_domain_probe):
    csv_file = temp_results_path / f"viscoelastic_simple_shear_{case}_center.csv"
    data, _ = read_domain_probe(csv_file)
    return data["Time"].to_numpy(), data["stress_xy"].to_numpy()


def relative_errors(sig_num, sig_ana):
    """MAE / mean|sigma| and RMSE / rms(sigma): global normalisers, because sigma
    crosses zero (sinusoid, after a drop) and a pointwise ratio is unbounded there."""
    diff = np.abs(sig_num - sig_ana)
    rel_mae = diff.mean() / np.abs(sig_ana).mean()
    rel_rmse = np.sqrt((diff**2).mean()) / np.sqrt((sig_ana**2).mean())
    return rel_mae, rel_rmse


@pytest.mark.parametrize("case", CASES)
def test_time_step(case, temp_results_path, read_tandem_summary, viscoelastic_simple_shear_config):
    """
    The step is capped at theta * tau and nothing else limits it here, so every
    step but the last is exactly theta * tau. The tolerances below assume the
    theta of simple_shear.lua; this fails first if that changes.
    """
    conf = viscoelastic_simple_shear_config
    summary = read_tandem_summary(temp_results_path / f"viscoelastic_simple_shear_{case}.log")
    tau = conf["eta"] / conf["mu1"]
    expected = conf["theta"] * tau
    assert summary["max_time_step"] == pytest.approx(expected, rel=conf["rel_dt"]), (
        f"max_time_step {summary['max_time_step']:.10e} != theta * tau = {expected:.10e}"
    )
    assert summary["step_rejections"] == 0


@pytest.mark.parametrize("case", CASES)
def test_initial_stress(case, temp_results_path, read_domain_probe, viscoelastic_simple_shear_config):
    """
    sigma_xy(0) must be the unrelaxed response (mu0 + mu1) gamma(0): at t = 0 the
    dashpot cannot move. Guards the unrelaxed t = 0 solve and q(0) = eps(0) in
    SeasQDOperator::initial_condition and the t = 0 stress output.
    """
    conf = viscoelastic_simple_shear_config
    model = SimpleShearSLS(conf)
    t, sig = load_case(case, temp_results_path, read_domain_probe)
    assert t[0] == 0.0, f"first sample at t = {t[0]}, expected t = 0"

    scale = (model.mu0 + model.mu1) * model.E0
    expected = (model.mu0 + model.mu1) * float(model.strain(case, np.array([0.0]))[0])
    err = abs(sig[0] - expected) / scale
    assert err < conf["sigma0_rel"], (
        f"sigma_xy(0) = {sig[0]:.16e}, expected (mu0 + mu1) gamma(0) = {expected:.16e} "
        f"(relative error {err:.2e} >= {conf['sigma0_rel']:.0e})"
    )


@pytest.mark.parametrize("case", CASES)
def test_stress_history(case, temp_results_path, read_domain_probe, viscoelastic_simple_shear_config):
    """
    Relative MAE and RMSE of sigma_xy(t) against the closed form. The two cases
    with a piecewise linear strain history must be exact to roundoff; the
    others must stay within headroom * the measured error (see conftest).
    """
    conf = viscoelastic_simple_shear_config
    model = SimpleShearSLS(conf)
    t, sig = load_case(case, temp_results_path, read_domain_probe)
    rel_mae, rel_rmse = relative_errors(sig, model.stress(case, t))

    if case in EXACT_CASES:
        limits = {"rel_mae": conf["exact"], "rel_rmse": conf["exact"]}
    else:
        limits = {k: conf["headroom"] * v for k, v in conf["measured"][case].items()}

    assert rel_mae < limits["rel_mae"], (
        f"{case}: relative MAE {rel_mae:.4e} >= {limits['rel_mae']:.4e}"
    )
    assert rel_rmse < limits["rel_rmse"], (
        f"{case}: relative RMSE {rel_rmse:.4e} >= {limits['rel_rmse']:.4e}"
    )
