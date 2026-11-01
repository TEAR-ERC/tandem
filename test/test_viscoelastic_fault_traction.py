"""
Fault traction written after each step, for simple shear of a standard linear solid
across a fault that does not slip.

Runs examples/tandem/viscoelasticity/2d/fault_shear.{geo,lua,toml} through app/tandem
(QD), see scripts/2D/viscoelastic_fault_traction.sh: the antiplane variant
(PoissonViscoelasticity) with classic RK4 at a fixed step and with Dormand-Prince 5(4),
and the inplane variant (Viscoelasticity) with classic RK4. The box is under Dirichlet
control with a displacement linear in y, held from t = 0, and the fault is practically
locked, so the shear traction on the fault is the uniform shear stress

    tau(t) = (mu0 + mu1 exp(-t / tau_relax)) * gamma,

exact to roundoff at every step (constant strain, affine field).

The traction written by the monitor after a step must combine the accepted displacement
with the viscoelastic history of that step. Without FSAL (RK4) the post-step solves at the
accepted state and then commits the history; a traction evaluated after the commit adds
the history at the end of the step instead, and is off by
mu1 (1 - exp(-dt / tau_relax)) q, 2.4 % here. PoissonViscoelasticity reads that history
from the buffers the commit writes, so the antiplane RK4 case is the one that fails then.
With FSAL the last stage provides the traction and nothing is evaluated after the commit.
"""

import numpy as np
import pytest

WORK_DIR = "viscoelastic_fault_traction"
CASES = ["antiplane_rk4", "antiplane_5dp", "inplane_rk4"]
RK4_CASES = ["antiplane_rk4", "inplane_rk4"]


def load_case(case, temp_results_path, read_domain_probe):
    data, _ = read_domain_probe(temp_results_path / WORK_DIR / case / "fault_shear_mid.csv")
    return data


def exact_traction(t, conf):
    tau_relax = conf["viscosity"] / conf["mu1"]
    return (conf["mu0"] + conf["mu1"] * np.exp(-t / tau_relax)) * conf["gamma"]


@pytest.mark.parametrize("case", CASES)
def test_fault_traction(case, temp_results_path, read_domain_probe,
                        viscoelastic_fault_traction_config):
    conf = viscoelastic_fault_traction_config
    data = load_case(case, temp_results_path, read_domain_probe)
    t = data["Time"].to_numpy()
    traction = data["traction0"].to_numpy()
    assert t[0] == 0.0 and len(t) > 100

    # The sign of the shear traction depends on the fault orientation; t = 0 fixes it.
    traction = np.sign(traction[0]) * traction
    exact = exact_traction(t, conf)
    err = np.abs(traction - exact) / exact
    worst = int(np.argmax(err))
    assert err[worst] < conf["exact"], (
        f"{case}: traction {traction[worst]:.16e} at t = {t[worst]:.6e}, expected "
        f"{exact[worst]:.16e} (relative error {err[worst]:.2e} >= {conf['exact']:.0e})"
    )


@pytest.mark.parametrize("case", CASES)
def test_fault_locked(case, temp_results_path, read_domain_probe,
                      viscoelastic_fault_traction_config):
    """The closed form assumes no slip; guard the friction setup that ensures it."""
    conf = viscoelastic_fault_traction_config
    data = load_case(case, temp_results_path, read_domain_probe)
    max_slip = data["slip0"].abs().max()
    assert max_slip < conf["max_slip"], f"{case}: slip {max_slip:.2e} m"


@pytest.mark.parametrize("case", RK4_CASES)
def test_linear_solves(case, temp_results_path, read_tandem_summary):
    """
    Without FSAL each step solves once per RK stage and once in the post-step, at the
    accepted state; the monitor reuses that solve and its traction. Two solves for the
    initial condition and one for the output at t = 0 come on top.
    """
    summary = read_tandem_summary(temp_results_path / f"viscoelastic_fault_traction_{case}.log")
    steps = summary["time_steps"]
    assert summary["step_rejections"] == 0
    expected = 3 + 5 * steps
    assert summary["linear_solves"] == expected, (
        f"linear_solves = {summary['linear_solves']:.0f}, expected 3 + 5 x {steps:.0f} steps "
        f"= {expected:.0f}"
    )
