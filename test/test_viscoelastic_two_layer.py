"""
Creep of two stacked standard linear solids with different relaxation times
(tau = 1 below y = 0.5, tau = 10 above) under the same uniform shear stress.

Runs examples/tandem/viscoelasticity/2d/two_layer.{geo,lua,toml} through app/tandem
(QD, no fault), see scripts/2D/viscoelastic_two_layer.sh. Each layer must relax at its
own rate: the relaxation time enters g(dt/tau) and exp(-dt/tau) at every quadrature
point, and only the step cap uses the smallest one. With the smallest tau everywhere
the upper layer creeps too fast and the upper probe is 12 % off.

Closed form (each layer an SLS in shear under constant stress s):

    gamma(t; tau) = s/mu0 - (s/mu0 - s/(mu0 + mu1)) exp(-t / tau_c),
    tau_c = tau (mu0 + mu1) / mu0,

    u_x(y, t) = gamma(t; tau_low) y                                      for y < 0.5,
    u_x(y, t) = gamma(t; tau_low) 0.5 + gamma(t; tau_high) (y - 0.5)     for y > 0.5,

and u_y = 0, sigma_xy = s, all other stress components 0. The field is piecewise linear
in y and the mesh conforms to y = 0.5, so the run is exact in space.

The second run gives a deprecated relaxation_time that contradicts viscosity / mu1 in
the upper layer; tandem must stop with a message instead of using either value.
"""

import numpy as np
import pytest

PROBES = {"low": 0.25, "high": 0.75}
WORK_DIR = "viscoelastic_two_layer"


def creep_strain(t, tau, conf):
    mu0, mu1, s = conf["mu0"], conf["mu1"], conf["traction"]
    tau_c = tau * (mu0 + mu1) / mu0
    return s / mu0 - (s / mu0 - s / (mu0 + mu1)) * np.exp(-t / tau_c)


def exact_ux(t, y, conf):
    low = creep_strain(t, conf["tau_low"], conf)
    if y < conf["interface"]:
        return low * y
    high = creep_strain(t, conf["tau_high"], conf)
    return low * conf["interface"] + high * (y - conf["interface"])


@pytest.fixture(scope="module")
def probes(temp_results_path, read_domain_probe):
    return {
        name: read_domain_probe(temp_results_path / WORK_DIR / f"two_layer_{name}.csv")
        for name in PROBES
    }


def test_time_step(temp_results_path, read_tandem_summary, viscoelastic_two_layer_config):
    """Every step is theta * tau_min, and the operator is rebuilt only for t = 0 and back."""
    conf = viscoelastic_two_layer_config
    summary = read_tandem_summary(temp_results_path / "viscoelastic_two_layer.log")
    expected = conf["theta"] * min(conf["tau_low"], conf["tau_high"])
    for key in ("min_time_step", "max_time_step"):
        assert summary[key] == pytest.approx(expected, rel=conf["rel_dt"]), (
            f"{key} {summary[key]:.10e} != theta * tau_min = {expected:.10e}"
        )
    assert summary["time_steps"] == round(conf["final_time"] / expected)
    assert summary["operator_reassemblies"] == 2


@pytest.mark.parametrize("name", PROBES)
def test_creep(name, probes, viscoelastic_two_layer_config):
    """u_x at each probe follows the creep of the layers below it, each at its own rate."""
    conf = viscoelastic_two_layer_config
    data, x = probes[name]
    assert x[1] == pytest.approx(PROBES[name]), f"probe {name} evaluated at {x}"
    t = data["Time"].to_numpy()
    ux = data["u0"].to_numpy()
    exact = exact_ux(t, x[1], conf)
    scale = np.max(np.abs(exact))

    assert abs(ux[0] / exact[0] - 1.0) < conf["exact"], (
        f"u_x(0) = {ux[0]:.10e}, expected the unrelaxed response {exact[0]:.10e}"
    )
    err = np.max(np.abs(ux - exact)) / scale
    limit = conf["headroom"] * conf["measured"][name]
    assert err < limit, f"probe {name}: max |u_x - exact| / max |u_x| = {err:.4e} >= {limit:.4e}"
    uy = np.max(np.abs(data["u1"].to_numpy())) / scale
    assert uy < conf["exact"], f"probe {name}: max |u_y| / max |u_x| = {uy:.3e}"


@pytest.mark.parametrize("name", PROBES)
def test_stress(name, probes, viscoelastic_two_layer_config):
    """The stress output is the uniform shear stress in both layers at all times."""
    conf = viscoelastic_two_layer_config
    data, _ = probes[name]
    s = conf["traction"]
    sxy = np.max(np.abs(data["stress_xy"].to_numpy() / s - 1.0))
    assert sxy < conf["exact"], f"probe {name}: max |stress_xy / s - 1| = {sxy:.3e}"
    for comp in ("stress_xx", "stress_yy"):
        other = np.max(np.abs(data[comp].to_numpy())) / s
        assert other < conf["exact"], f"probe {name}: max |{comp}| / s = {other:.3e}"


def test_inconsistent_relaxation_time_rejected(temp_results_path):
    """A relaxation_time that contradicts viscosity / mu1 stops the run with a message."""
    log = temp_results_path / "viscoelastic_two_layer_inconsistent.log"
    text = log.read_text(encoding="utf-8")
    assert "relaxation_time disagrees with viscosity / mu1" in text, (
        f"the inconsistent run was not rejected; see {log}"
    )
    assert "========= Summary =========" not in text, "the inconsistent run ran to the end"
