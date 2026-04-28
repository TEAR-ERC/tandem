"""
Right-hand side (RHS) assembly kernels for viscoelasticity.

Generates the YaTeTo kernels for the linear form (RHS vector b).
This includes standard external forcing, lifting operators for boundary conditions,
Neumann/Dirichlet/Free-slip surface integrals, and the viscoelastic history source terms.

Notes:
-----
The history terms represent the memory of the Standard Linear Solid model:
    Volume:  -∫_E ε_ij(v) σ̂ⁿ_ij dE
    Surface: +∫_∂E [[v_i]] {σ̂ⁿ_ij n_j} dS
Where σ̂ⁿ is the history stress derived from the partial strain qⁿ and
the deviatoric strain ε^{dev,n} from the previous time step.
"""


def add(generator, t):
    generator.add(
        "rhs_lift_ip",
        # IP lifting of boundary traction: f_lifted_q = f_q * |n|
        t.f_lifted_q["iq"] <= t.f_q["iq"] * t.nl_q["q"],
    )

    generator.add(
        "rhs_lift_boundary",
        # BR2 lifting at boundary: f_lifted[0] then project via test_normal
        [
            t.f_lifted[0]["ums"]
            <= t.Minv[0]["mo"] * t.w["q"] * t.E_q[0]["oq"] * t.f_q["uq"] * t.n_q["sq"],
            t.f_lifted_q["iq"]
            <= t.test_normal(0) * t.E_q[0]["mq"] * t.f_lifted[0]["ums"],
        ],
    )

    generator.add(
        "rhs_lift_skeleton",
        # BR2 lifting at interior facets: average over both sides
        [
            t.f_lifted[0]["ums"]
            <= 0.5
            * t.Minv[0]["mo"]
            * t.w["p"]
            * t.E_q[0]["op"]
            * t.f_q["up"]
            * t.n_q["sp"],
            t.f_lifted[1]["ums"]
            <= 0.5
            * t.Minv[1]["mo"]
            * t.w["p"]
            * t.E_q[1]["op"]
            * t.f_q["up"]
            * t.n_q["sp"],
            t.f_lifted_q["iq"]
            <= 0.5
            * (
                t.test_normal(0) * t.E_q[0]["mq"] * t.f_lifted[0]["ums"]
                + t.test_normal(1) * t.E_q[1]["mq"] * t.f_lifted[1]["ums"]
            ),
        ],
    )

    generator.add(
        "rhsVolumeForce",
        # ∫_E f_i v_i |J| W dξ
        t.b["kp"] <= (t.b["kp"] + t.J["q"] * t.W["q"] * t.E_Q["kq"] * t.F_Q["pq"]),
    )

    generator.add(
        "rhsFacet",
        # Standard SIPG Dirichlet/fault RHS boundary term:
        # b[kp] += c1 σ̂ⁿ⁺¹(v, f) w + c2 w φ_k f_lifted
        # First line recomputes Dx_q[0] (physical gradient) from metric g[0]
        [
            t.Dx_q[0]["kiq"] <= t.g[0]["eiq"] * t.Dxi_q[0]["keq"],
            t.b["kp"]
            <= (
                t.b["kp"]
                + t.c1[0] * t.tractionTest(0, t.f_q) * t.w["q"]
                + t.c2[0] * t.w["q"] * t.E_q[0]["kq"] * t.f_lifted_q["pq"]
            ),
        ],
    )

    generator.add(
        "rhsFreeSlip",
        # Free-slip BC: only normal stress symmetry and normal penalty terms.
        # g_q["q"] is the prescribed normal displacement (scalar BC value).
        # No consistency term (no Neumann flux for tangential directions).
        [
            t.Dx_q[0]["kiq"] <= t.g[0]["eiq"] * t.Dxi_q[0]["keq"],
            t.b["kp"]
            <= (
                t.b["kp"]
                # Symmetry: g * σ̂_nn(v) * |n|
                + t.c1[0] * t.w["q"] * t.g_q["q"] * t.normalStressTest(0) * t.nl_q["q"]
                # Penalty: g * (v·n̂) * |n|
                + t.c2[0]
                * t.w["q"]
                * (t.E_q[0]["kq"] * t.n_unit_q["pq"])  # v · n̂
                * t.g_q["q"]
                * t.nl_q["q"]  # g * area
            ),
        ],
    )

    generator.add(
        "rhsTraction",
        # Neumann (traction) BC: ∫_Γ t_p v_k |n| w dξ
        # traction_component["pq"] is the prescribed traction vector at facet pts.
        [
            t.b["kp"]
            <= t.b["kp"]
            + t.w["q"] * t.E_q[0]["kq"] * t.traction_component["pq"] * t.nl_q["q"]
        ],
    )

    # HISTORY RHS KERNELS (new VE-specific RHS terms)
    #
    # From the weak form (derived in the PDF), the NEW RHS terms are:
    #
    #   -∫_E ε_ij(v) σ̂ⁿ_ij dE  +  ∫_∂E [[v_i]] {σ̂ⁿ_ij n_j} dS
    #
    # where σ̂ⁿ_ij = 2μ₁ [exp(-Δt/τ) qⁿ_ij - g(Δt) ε^{dev,n}_ij]
    #
    # The volume term uses volume quad pts (Q); the surface terms use facet
    # quad pts (q). Both reference the OLD (time n) strain quantities.

    generator.add(
        "rhsVolumeHistory",
        # -∫_E ε_ij(v) σ̂ⁿ_ij dE
        #
        # Expanding σ̂ⁿ:
        #   = -∫_E ε_ij(v) 2μ₁ [ratio qⁿ_ij - g_dt ε^{dev,n}_ij] dE
        #   = +∫_E ε_ij(v) 2μ₁ g_dt ε^{dev,n}_ij dE
        #     -∫_E ε_ij(v) 2μ₁ ratio qⁿ_ij dE
        #
        # Discrete (basis fn k, component p, quad pt q contracted on i,j):
        #   b[kp] += 2 μ₁_W_J_Q[q] g_dt_Q[q] ε^{dev,n}_pj[q] Dx_Q[kjq]
        #           -2 μ₁_W_J_Q[q] ratio_Q[q]  qⁿ_pj[q]       Dx_Q[kjq]
        #
        # Layout note: deviatoric_strain_tensor_Q["qpj"] = ε^{dev}[quad, p, j]
        #              partial_strain_tensor_Q["qpj"] = qⁿ[quad, p, j]
        t.b["kp"]
        <= (
            t.b["kp"]
            + 2.0
            * (
                t.g_dt_Q["q"]
                * t.deviatoric_strain_tensor_Q["qpj"]
                * t.mu1_W_J_Q["q"]
                * t.Dx_Q["kjq"]
                - t.partial_strain_tensor_Q["qpj"]
                * t.mu1_W_J_Q["q"]
                * t.ratio_Q["q"]
                * t.Dx_Q["kjq"]
            )
        ),
    )

    generator.add(
        "tractionHistoryAverage",
        # Precompute average history traction across interior facet:
        #   {σ̂ⁿ n}_p = 0.5 * Σ_{side} (μ₁ ratio qⁿ_pj n_j - μ₁ g_dt ε^{dev,n}_pj n_j)
        #
        # tractionHistory(side, t.n_q) returns:
        #   μ₁_q[side] ratio_q qⁿ_pj n_j  -  μ₁_q[side] g_dt_q ε^{dev,n}_pj n_j
        # (the factor of 2 from σ̂ⁿ = 2μ₁[...] is applied in rhsSurface via c3)
        [
            t.tractionHistoryAverage["qp"]
            <= 0.5 * (t.tractionHistory(0, t.n_q) + t.tractionHistory(1, t.n_q))
        ],
    )

    generator.add(
        "rhsSurface",
        # Interior facet history surface term:
        #   + ∫_∂E_int [[v_i]] {σ̂ⁿ_ij n_j} dS
        #
        # [[v_i]] for the minus side (side 0): v⁻_i.
        # The average is tractionHistoryAverage["qp"] (precomputed above).
        # c3 absorbs the factor of 2 from σ̂ⁿ = 2μ₁[...].
        [
            t.b["kp"]
            <= t.b["kp"]
            + t.c3 * t.tractionHistoryAverage["qp"] * t.E_q[0]["kq"] * t.w["q"]
        ],
    )

    generator.add(
        "rhsSurfaceNormal",
        # Free-slip history surface term (normal projection only):
        #   + ∫_∂E_freeslip (v·n̂) {σ̂ⁿ_nn} dS
        #
        # {σ̂ⁿ_nn} = tractionHistoryAverage · n̂  (scalar normal stress)
        # v·n̂      = E_q[0]["kq"] * n_unit_q["pq"]
        [
            # Project average history traction onto unit normal
            # tractionHistoryNormal[q] = {σ̂ⁿ_pj n_j}_{avg} · n̂_p
            t.tractionHistoryNormal["q"]
            <= t.tractionHistoryAverage["qp"] * t.n_unit_q["pq"],
            # Test against normal component of v
            t.b["kp"]
            <= t.b["kp"]
            + t.c3
            * t.w["q"]
            * (t.E_q[0]["kq"] * t.n_unit_q["pq"])  # v · n̂
            * t.tractionHistoryNormal["q"],  # {σ̂ⁿ_nn}
        ],
    )
