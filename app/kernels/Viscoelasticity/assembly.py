"""
Assembly kernels for the viscoelastic stiffness matrices.

Generates the YaTeTo kernels for the discrete bilinear form.
This includes the volume stiffness matrix (A) and the Discontinuous Galerkin (SIPG)
surface integrals (consistency, symmetry, and penalty terms) for standard interior/fault
facets as well as free-slip boundaries.

"""

from yateto import simpleParameterSpace


def add(generator, t):
    generator.add(
        "assembleVolume",
        # Bilinear form ∫_E ε_ij(v) Ĉ_ijkl ε_kl(u) dE discretised:
        # A[kp,lu] = A_dt * ∂φ_l/∂x_u * ∂φ_k/∂x_p
        #           + B_dt * ∂φ_k/∂x_j (∂φ_l/∂x_j δ_{pu} + ∂φ_l/∂x_p δ_{ju})
        # (W*J already folded into A_dt, B_dt via precomputeVolumeAB)
        t.A_mat["kplu"]
        <= (
            t.A_dt["q"] * t.Dx_Q["luq"] * t.Dx_Q["kpq"]
            + t.B_dt["q"]
            * t.Dx_Q["kjq"]
            * (t.Dx_Q["ljq"] * t.delta["pu"] + t.Dx_Q["lpq"] * t.delta["ju"])
        ),
    )

    generator.addFamily(
        "assembleTractionOp",
        simpleParameterSpace(2),
        lambda x: t.assembleTractionOp(x),
    )

    generator.addFamily(
        "assembleSurface",
        simpleParameterSpace(2, 2),
        # SIPG surface bilinear form for interior/Dirichlet/fault facets:
        # a[x,y][kp,lu] = c0[x] φ_k^x T^y_{lupq}      [consistency]
        #               + c1[y] φ_l^y T^x_{kpuq}       [symmetry]
        #               + c2    φ_k^x L^y_{lpuq}       [penalty]
        lambda x, y: t.a[x][y]["kplu"]
        <= (
            t.c0[x] * t.E_q[x]["kq"] * t.w["q"] * t.traction_op_q[y]["lupq"]
            + t.c1[y] * t.E_q[y]["lq"] * t.w["q"] * t.traction_op_q[x]["kpuq"]
            + t.c2[abs(y - x)] * t.w["q"] * t.E_q[x]["kq"] * t.L_q[y]["lpuq"]
        ),
    )

    generator.addFamily(
        "assembleSurfaceFreeSlip",
        simpleParameterSpace(2, 2),
        # Free-slip surface bilinear form (only normal component is constrained):
        # Consistency: -(v·n̂)(σ̂ⁿ⁺¹(u)·n̂)     [projects traction onto normal]
        # Symmetry:    -(u·n̂)(σ̂ⁿ⁺¹(v)·n̂)|n|   [σ_nn test, unscaled → needs nl_q]
        # Penalty:     +γ (u·n̂)(v·n̂)|n|        [normal-only penalty]
        lambda x, y: t.a[x][y]["kplu"]
        <= (
            t.c0[x]
            * t.w["q"]
            * (t.E_q[x]["kq"] * t.n_unit_q["pq"])
            * (t.traction_op_q[y]["lusq"] * t.n_unit_q["sq"])
            + t.c1[y]
            * t.w["q"]
            * t.normalStressTest(x)
            * (t.E_q[y]["lq"] * t.n_unit_q["uq"])
            * t.nl_q["q"]
            + t.c2[abs(y - x)]
            * t.w["q"]
            * (t.E_q[x]["kq"] * t.n_unit_q["pq"])
            * (t.L_q[y]["lsuq"] * t.n_unit_q["sq"])
        ),
    )
