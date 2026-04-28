"""
Traction evaluation kernels for viscoelasticity.

Generates the YaTeTo kernels for evaluating the numerical
traction vectors at element interfaces. These kernels compute both current-step
and history traction contributions for interior facets and boundaries/faults.
"""


def add(generator, t):
    generator.add(
        "compute_traction",
        # Interior facet traction (average of both sides + penalty):
        #   t_p = 0.5[(σ̂ⁿ⁺¹(u⁻)+σ̂ⁿ(u⁻))·n̂ + (σ̂ⁿ⁺¹(u⁺)+σ̂ⁿ(u⁺))·n̂] + c0 [[u_p]]
        # tractionHistory() omits the factor 2 from σ̂ⁿ, so it is applied explicitly.
        t.traction_q["pq"]
        <= 0.5
        * (
            t.traction(0, t.n_unit_q)
            + 2.0 * t.tractionHistory(0, t.n_unit_q)
            + t.traction(1, t.n_unit_q)
            + 2.0 * t.tractionHistory(1, t.n_unit_q)
        )
        + t.c0[0]
        * (t.E_q[0]["lq"] * t.u[0]["lp"] - t.E_q[1]["lq"] * t.u[1]["lp"] - t.f_q["pq"]),
    )

    generator.add(
        "compute_traction_bnd",
        # Boundary/fault traction (one-sided + penalty):
        #   t_p = (σ̂ⁿ⁺¹(u⁻)+σ̂ⁿ(u⁻))·n̂ + c0 (u⁻_p - f_p)
        # tractionHistory() omits the factor 2 from σ̂ⁿ, so it is applied explicitly.
        t.traction_q["pq"]
        <= t.traction(0, t.n_unit_q)
        + 2.0 * t.tractionHistory(0, t.n_unit_q)
        + t.c0[0] * (t.E_q[0]["lq"] * t.u[0]["lp"] - t.f_q["pq"]),
    )
