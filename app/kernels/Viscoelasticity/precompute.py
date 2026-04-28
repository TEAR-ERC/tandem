"""
Setup-phase YaTeTo kernels for material projection and lifting.

Contains kernels that are executed once per element/facet during the
initialization phase (e.g., mass matrices, physical gradients, lifting
operators) and time-dependent effective Lamé parameter updates executed
once per time step.
"""

from yateto import simpleParameterSpace


def add(generator, t):
    """
    Register all precomputation kernels with the YaTeTo generator.

    Parameters
    ----------
    generator : yateto.Generator
    t         : SimpleNamespace returned by make_tensors()
    """

    # MATERIAL PROJECTION
    generator.add(
        "massMatrix",
        # M_kl = ∫_E φ_k φ_l W |J| dξ
        t.M["kl"] <= t.E_Q["kq"] * t.W["q"] * t.J["q"] * t.E_Q["lq"],
    )

    generator.add(
        "project_material_lhs",
        # Material mass matrix for L2 projection (uses matE_Q_T layout)
        t.matM["kl"] <= t.matE_Q_T["qk"] * t.W["q"] * t.J["q"] * t.matE_Q_T["ql"],
    )

    generator.add(
        "project_material_rhs",
        # Project each material property from quad points onto nodal basis:
        # prop["k"] = Σ_q prop_Q["q"] * matE_Q_T["qk"] * W["q"] * J["q"]
        [
            t.lam["k"] <= t.lam_Q["q"] * t.matE_Q_T["qk"] * t.W["q"] * t.J["q"],
            t.mu0["k"] <= t.mu0_Q["q"] * t.matE_Q_T["qk"] * t.W["q"] * t.J["q"],
            t.mu1["k"] <= t.mu1_Q["q"] * t.matE_Q_T["qk"] * t.W["q"] * t.J["q"],
            t.viscosity["k"]
            <= t.viscosity_Q["q"] * t.matE_Q_T["qk"] * t.W["q"] * t.J["q"],
            t.rhoInv["k"] <= t.rhoInv_Q["q"] * t.matE_Q_T["qk"] * t.W["q"] * t.J["q"],
        ],
    )

    # VOLUME PRECOMPUTATION
    generator.add(
        "precomputeVolume",
        # Fold W*J into material arrays once so every kernel avoids
        # repeating the multiply:
        #   lam_W_J_Q["q"] = (Σ_t matE_Q_T["qt"] lam["t"]) * W["q"] * J["q"]
        [
            t.lam_W_J_Q["q"] <= t.matE_Q_T["qt"] * t.lam["t"] * t.W["q"] * t.J["q"],
            t.mu0_W_J_Q["q"] <= t.matE_Q_T["qt"] * t.mu0["t"] * t.W["q"] * t.J["q"],
            t.mu1_W_J_Q["q"] <= t.matE_Q_T["qt"] * t.mu1["t"] * t.W["q"] * t.J["q"],
            t.viscosity_W_J_Q["q"]
            <= t.matE_Q_T["qt"] * t.viscosity["t"] * t.W["q"] * t.J["q"],
            t.negative_rhoInv_W_Jinv_Q["q"]
            <= -t.matE_Q_T["qt"] * t.rhoInv["t"] * t.W["q"] * t.Jinv_Q["q"],
        ],
    )

    generator.add(
        "precomputeVolumeAB",
        # Effective Lamé params at volume quad pts, W*J-scaled, from g(Δt):
        #   A(Δt) = lam_W_J + (2/3) mu1_W_J (1 - g_dt)
        #   B(Δt) = mu0_W_J + mu1_W_J * g_dt
        # Note: 2/3 approximated as 0.67 matching the reference implementation.
        [
            t.A_dt["q"]
            <= (
                t.lam_W_J_Q["q"]
                + 0.67 * (t.mu1_W_J_Q["q"] - t.g_dt_Q["q"] * t.mu1_W_J_Q["q"])
            ),
            t.B_dt["q"] <= t.mu0_W_J_Q["q"] + t.mu1_W_J_Q["q"] * t.g_dt_Q["q"],
        ],
    )

    generator.add(
        "precomputeVolumeABUnscaled",
        # Unscaled effective Lamé params for stress output (not folded with W*J):
        #   A(Δt) = λ + (2/3) μ₁ (1 - g(Δt))
        #   B(Δt) = μ₀ + μ₁ g(Δt)
        [
            t.A_dt_unscaled["q"]
            <= (t.lam_Q["q"] + 0.67 * (t.mu1_Q["q"] - t.g_dt_Q["q"] * t.mu1_Q["q"])),
            t.B_dt_unscaled["q"] <= t.mu0_Q["q"] + t.mu1_Q["q"] * t.g_dt_Q["q"],
        ],
    )

    generator.add(
        "Dx_Q",
        # Physical gradient of basis functions at volume quad pts:
        #   ∂φ_k/∂x_i = G_{ei} ∂φ_k/∂ξ_e
        t.Dx_Q["kiq"] <= t.G["eiq"] * t.Dxi_Q["keq"],
    )

    # FACET PRECOMPUTATION
    generator.addFamily(
        "precomputeSurface",
        simpleParameterSpace(2),
        # Interpolate nodal material values to facet quad pts for side x:
        #   prop_q[x]["q"] = Σ_t matE_q_T[x]["qt"] prop["t"]
        lambda x: [
            t.lam_q[x]["q"] <= t.matE_q_T[x]["qt"] * t.lam["t"],
            t.mu0_q[x]["q"] <= t.matE_q_T[x]["qt"] * t.mu0["t"],
            t.mu1_q[x]["q"] <= t.matE_q_T[x]["qt"] * t.mu1["t"],
            t.viscosity_q[x]["q"] <= t.matE_q_T[x]["qt"] * t.viscosity["t"],
        ],
    )

    generator.addFamily(
        "precomputeSurfaceAB",
        simpleParameterSpace(2),
        # Effective Lamé params at facet quad pts from g_dt_q:
        #   A(Δt) = lam_q + (2/3) mu1_q (1 - g_dt_q)
        #   B(Δt) = mu0_q + mu1_q * g_dt_q
        lambda x: [
            t.A_dt_q[x]["q"]
            <= (
                t.lam_q[x]["q"]
                + 0.67 * (t.mu1_q[x]["q"] - t.mu1_q[x]["q"] * t.g_dt_q["q"])
            ),
            t.B_dt_q[x]["q"] <= t.mu0_q[x]["q"] + t.mu1_q[x]["q"] * t.g_dt_q["q"],
        ],
    )

    generator.addFamily(
        "Dx_q",
        simpleParameterSpace(2),
        # Physical gradient of basis functions at facet quad pts for side x:
        #   ∂φ_k/∂x_i = g[x]_{ei} ∂φ_k/∂ξ_e
        lambda x: t.Dx_q[x]["kiq"] <= t.g[x]["eiq"] * t.Dxi_q[x]["keq"],
    )

    # LIFT OPERATORS
    # IP (interior penalty): cheap, no mass matrix inverse.
    # BR2: uses local mass matrix inverse, more accurate for curved elements.

    generator.addFamily(
        "lift_ip",
        simpleParameterSpace(2),
        # IP lifting: L_q[x]["lpuq"] = φ_l(x_q) δ_{pu} |n|_q
        lambda x: t.L_q[x]["lpuq"] <= t.E_q[x]["lq"] * t.delta["pu"] * t.nl_q["q"],
    )

    generator.addFamily(
        "lift_skeleton",
        simpleParameterSpace(2),
        # BR2 lifting for interior facets (averaged over both sides):
        #   Lift[s]["lsm"] = 0.5 * Minv[s]_{mo} ∫ φ_o φ_l^x n_s w dξ
        lambda x: [
            t.Lift[0]["lsm"]
            <= (
                0.5
                * t.Minv[0]["mo"]
                * t.w["q"]
                * t.E_q[0]["oq"]
                * t.E_q[x]["lq"]
                * t.n_q["sq"]
            ),
            t.Lift[1]["lsn"]
            <= (
                0.5
                * t.Minv[1]["no"]
                * t.w["q"]
                * t.E_q[1]["oq"]
                * t.E_q[x]["lq"]
                * t.n_q["sq"]
            ),
            t.L_q[x]["liuq"]
            <= (
                0.5
                * (
                    t.test_normal(0) * t.E_q[0]["mq"] * t.Lift[0]["lsm"]
                    + t.test_normal(1) * t.E_q[1]["nq"] * t.Lift[1]["lsn"]
                )
            ),
        ],
    )

    generator.add(
        "lift_boundary",
        # BR2 lifting for boundary facets (one-sided, no factor 0.5):
        [
            t.Lift[0]["lsm"]
            <= (
                t.Minv[0]["mo"]
                * t.w["q"]
                * t.E_q[0]["oq"]
                * t.E_q[0]["lq"]
                * t.n_q["sq"]
            ),
            t.L_q[0]["liuq"]
            <= (
                t.test_normal(0) * t.E_q[0]["mq"] * t.Lift[0]["lsm"]
            ),
        ],
    )
