"""
Creep and recovery of the viscoelastic rod under a traction history, compared
against the Zener uniaxial closed form (3D).

Runs examples/tandem/viscoelasticity/3d/cantilever_rod.{geo,lua,toml} through
app/tandem (QD, no fault), see scripts/3D/viscoelastic_cantilever.sh. The
traction s0 is applied on x = L at t = 0 and removed after t1.

Boundary conditions. The example's default (clamped = 1, the manuscript's
setup) clamps x = 0. A clamp blocks the lateral contraction, so the stress is
uniaxial only away from it and the closed form holds only approximately, with
an error that mixes the mesh and the time step. The test meshes the same .geo
with clamped = 0 instead: x = 0, y = 0 and z = 0 become symmetry planes (free
slip with u.n = 0, tag 9). The rod is then in exactly uniaxial stress, the
displacement u = (eps_xx x, eps_yy y, eps_yy z) is affine and lies in the
discrete space, and what is measured is the time integration under traction
control and the t = 0 solve alone, on a 42-cell mesh. It also exercises free
slip with the viscoelastic operator in 3D, the condition the Savage-Prescott
example relies on.

Strain measurement. Domain probes are evaluated at the nearest point of a
per-element sampling lattice, not at the requested position, and the header
prints that position with 6 significant digits. Because u_x = eps_xx x exactly,
eps_xx = u_x / x at any probe. That is used once, at t = 0, for the absolute
check of the unrelaxed solve (good to the 6-digit position, far below the
4.5% the check has to resolve). The history uses u_x(t) / u_x(0) * s0 / E0,
which does not depend on the position at all. The stress output is checked
separately (test_stress_output); it is not used to measure strain.

Closed form (manuscript, Sec. 4.2): with K the bulk modulus, G0 = mu0 + mu1,
E0 = 9 K G0 / (3 K + G0), Einf = 9 K mu0 / (3 K + mu0), tau_c = tau G0 / mu0,

    D(t)      = 1/Einf - (E0 - Einf) / (E0 Einf) exp(-t / tau_c),
    eps_xx(t) = s0 D(t)                  for t <= t1,
    eps_xx(t) = s0 (D(t) - D(t - t1))    for t > t1.

The load is on at t = t1 (lua: t <= T_load); the sample nearest t1 lands just
after it, because the step is theta * tau with tau = eta / mu1 a hair above
100 s. Comparing the CSV time (printed to 16 digits) against t1 with the same
<= reproduces the lua's choice of branch exactly.
"""

import math

import numpy as np
import pytest

PROBES = ["free_end", "mid", "near_fixed_end"]
WORK_DIR = "viscoelastic_cantilever"


class ZenerRod:
    """Uniaxial creep-recovery of the rod, see the module docstring."""

    def __init__(self, conf):
        g0, r1 = conf["g0"], conf["r1"]
        self.mu0 = (1.0 - r1) * g0
        self.mu1 = r1 * g0
        self.g0 = g0
        self.tau = conf["eta"] / self.mu1
        self.K = conf["bulk_modulus"]
        self.s0 = conf["traction"]
        self.t1 = conf["t_load"]
        self.E0 = 9.0 * self.K * g0 / (3.0 * self.K + g0)
        self.Einf = 9.0 * self.K * self.mu0 / (3.0 * self.K + self.mu0)
        self.tau_c = self.tau * g0 / self.mu0

    def compliance(self, t):
        return 1.0 / self.Einf - (self.E0 - self.Einf) / (self.E0 * self.Einf) * np.exp(
            -t / self.tau_c
        )

    def strain(self, t):
        t = np.asarray(t, float)
        eps = self.s0 * self.compliance(t)
        post = t > self.t1
        eps[post] = self.s0 * (self.compliance(t[post]) - self.compliance(t[post] - self.t1))
        return eps

    def rebound_excess(self, theta):
        """Strain overshoot of the unloading step, which sees B(dt) = mu0 + mu1 g(dt)."""
        g = -math.expm1(-theta) / theta
        B = self.mu0 + self.mu1 * g
        E_eff = 9.0 * self.K * B / (3.0 * self.K + B)
        return self.s0 * (1.0 / E_eff - 1.0 / self.E0)


@pytest.fixture(scope="module")
def rod(viscoelastic_cantilever_config):
    return ZenerRod(viscoelastic_cantilever_config)


@pytest.fixture(scope="module")
def probes(temp_results_path, read_domain_probe):
    return {
        name: read_domain_probe(
            temp_results_path / WORK_DIR / f"cantilever_rod_probe_{name}.csv"
        )
        for name in PROBES
    }


def strain_history(data, rod):
    """eps_xx(t) = u_x(t) / u_x(0) * s0 / E0: independent of the probe position."""
    u = data["u0"].to_numpy()
    return u / u[0] * rod.s0 / rod.E0


def test_time_step(temp_results_path, read_tandem_summary, viscoelastic_cantilever_config, rod):
    """Every step is theta * tau and the run reaches final_time."""
    conf = viscoelastic_cantilever_config
    summary = read_tandem_summary(temp_results_path / "viscoelastic_cantilever.log")
    expected = conf["theta"] * rod.tau
    for key in ("min_time_step", "max_time_step"):
        assert summary[key] == pytest.approx(expected, rel=conf["rel_dt"]), (
            f"{key} {summary[key]:.10e} != theta * tau = {expected:.10e}"
        )
    assert summary["time_steps"] == round(conf["final_time"] / expected)


@pytest.mark.parametrize("name", PROBES)
def test_initial_strain(name, probes, rod, viscoelastic_cantilever_config):
    """
    eps_xx(0) = s0 / E0 with the unrelaxed (instantaneous) modulus. This is the
    regression test for the unrelaxed t = 0 solve in
    SeasQDOperator::initial_condition: solving t = 0 with the step-sized moduli
    instead gives eps_xx(0) = 0.010450, 4.5% off, under traction control.
    Checked at three probes along the rod, which also checks that the state is
    homogeneous, i.e. that the symmetry planes and the traction-free sides act.
    """
    conf = viscoelastic_cantilever_config
    data, x = probes[name]
    assert data["Time"].iloc[0] == 0.0
    eps0 = data["u0"].iloc[0] / x[0]
    err = eps0 / (rod.s0 / rod.E0) - 1.0
    assert abs(err) < conf["eps0_rel"], (
        f"eps_xx(0) = {eps0:.10e} at probe {name} (x = {x}), expected s0/E0 = "
        f"{rod.s0 / rod.E0:.10e}: relative error {err:+.3e}"
    )


def test_creep(probes, rod, viscoelastic_cantilever_config):
    """Creep under constant traction, t <= t1, toward s0 / Einf with time constant tau_c."""
    conf = viscoelastic_cantilever_config
    data, _ = probes["free_end"]
    t = data["Time"].to_numpy()
    loaded = t <= rod.t1
    err = np.abs(strain_history(data, rod) - rod.strain(t))[loaded]

    limit = conf["headroom"] * conf["measured"]["creep_abs"]
    worst = int(np.argmax(err))
    assert err[worst] < limit, (
        f"creep: max |eps_xx - exact| = {err[worst]:.4e} at t = {t[loaded][worst]:.1f} s "
        f">= {limit:.4e}"
    )
    # The creep must actually have happened: eps_xx grows ~10x toward s0 / Einf.
    assert strain_history(data, rod)[loaded][-1] > 0.99 * rod.s0 / rod.Einf


def test_rebound(probes, rod, viscoelastic_cantilever_config):
    """
    The unloading step overshoots the elastic rebound by s0 (1/E_eff - 1/E0),
    a known first-order effect of the step that carries the load jump (see
    conftest). Bounded above, so that it passes today and fails if it doubles.
    """
    conf = viscoelastic_cantilever_config
    data, _ = probes["free_end"]
    t = data["Time"].to_numpy()
    after = np.flatnonzero(t > rod.t1)
    assert after.size, "no samples after the load is removed"
    i = after[0]
    err = strain_history(data, rod)[i] - rod.strain(t)[i]

    predicted = rod.rebound_excess(conf["theta"])
    limit = conf["headroom"] * predicted
    assert abs(err) < limit, (
        f"rebound at t = {t[i]!r}: eps_xx - exact = {err:.6e}, |.| >= {limit:.6e} "
        f"({abs(err) / predicted:.4f} x the predicted overshoot {predicted:.6e})"
    )


def test_recovery(probes, rod, viscoelastic_cantilever_config):
    """Recovery after unloading, t > t1, toward zero strain."""
    conf = viscoelastic_cantilever_config
    data, _ = probes["free_end"]
    t = data["Time"].to_numpy()
    post = t > rod.t1
    err = (strain_history(data, rod) - rod.strain(t))[post]
    rms = float(np.sqrt(np.mean(err**2)))

    limit = conf["headroom"] * conf["measured"]["recovery_rms"]
    assert rms < limit, f"recovery: RMS of eps_xx - exact = {rms:.4e} >= {limit:.4e}"


def test_stress_output(probes, rod, viscoelastic_cantilever_config):
    """
    While loaded, the exact stress is uniform uniaxial: stress_xx = s0, all
    other components 0. The probe stress is the output path (L2 projection of
    the stress rebuilt in Viscoelasticity::stress_volume), which the solve does
    not use, so a wrong output cannot show up in any displacement check. Under
    uniaxial stress tr(eps) != 0, so this checks the volumetric part of the
    output: leaving out the (2/3) mu1 tr(eps) I of the Maxwell spring makes
    stress_xx 0.67% low and stress_yy -6.7e-5 instead of 0.
    """
    conf = viscoelastic_cantilever_config
    for name in PROBES:
        data, _ = probes[name]
        loaded = data["Time"].to_numpy() <= rod.t1
        sxx = data["stress_xx"].to_numpy()[loaded]
        assert np.max(np.abs(sxx / rod.s0 - 1.0)) < conf["stress_rel"], (
            f"stress_xx at probe {name}: max relative error "
            f"{np.max(np.abs(sxx / rod.s0 - 1.0)):.3e}"
        )
        for comp in ("stress_yy", "stress_zz", "stress_xy", "stress_xz", "stress_yz"):
            other = data[comp].to_numpy()[loaded]
            assert np.max(np.abs(other)) / rod.s0 < conf["stress_rel"], (
                f"{comp} at probe {name}: max |.| / s0 = {np.max(np.abs(other)) / rod.s0:.3e}"
            )
