#!/usr/bin/env python3

from yateto import *


def add(generator, dim, nbf, Nbf, nq, Nq, petsc_alignment):
    J_Q = Tensor("J_Q", (Nq,))
    Jinv_Q = Tensor("Jinv_Q", (Nq,))
    G_Q = Tensor("G_Q", (dim, dim, Nq))
    K = Tensor("K", (Nbf,))  # mu_0
    MU1 = Tensor("mu1", (Nbf,))  # [VE] mu_1
    MU_EFF = Tensor("mu_eff", (Nbf,))  # [VE]
    GDT = Tensor("g_dt", (Nbf,))  # [VE] (tau/dt)(1 - exp(-dt/tau))
    RATIO = Tensor("ratio", (Nbf,))  # [VE] exp(-dt/tau)
    K_Q = Tensor("K_Q", (Nq,))
    W = Tensor("W", (Nq,))
    E_Q = Tensor("E_Q", (Nbf, Nq))
    matE_Q_T = Tensor("matE_Q_T", (Nq, Nbf))
    Dxi_Q = Tensor("Dxi_Q", (Nbf, dim, Nq))
    Dx_Q = Tensor("Dx_Q", Dxi_Q.shape())
    A = Tensor("A", (Nbf, Nbf))
    M = Tensor("M", (Nbf, Nbf))
    MinvRef = Tensor("MinvRef", (Nbf, Nbf))
    MinvWA = Tensor("MinvWA", (Nbf, Nbf))
    matM = Tensor("matM", (Nbf, Nbf))

    # [VE] Must be re-executed whenever dt changes, before any assembly.
    generator.add("get_mu_eff", MU_EFF["k"] <= K["k"] + GDT["k"] * MU1["k"])

    generator.add("massMatrix", M["kl"] <= E_Q["kq"] * W["q"] * J_Q["q"] * E_Q["lq"])
    generator.add(
        "MinvWA",
        MinvWA["kl"]
        <= MinvRef["kr"] * W["q"] * Jinv_Q["q"] * E_Q["rq"] * E_Q["sq"] * MinvRef["sl"],
    )

    # Reuse for mu_0, mu_1 and eta -- same lhs, different rhs data.
    generator.add(
        "project_K_lhs",
        matM["kl"] <= matE_Q_T["qk"] * W["q"] * J_Q["q"] * matE_Q_T["ql"],
    )
    generator.add(
        "project_K_rhs", K["k"] <= K_Q["q"] * matE_Q_T["qk"] * W["q"] * J_Q["q"]
    )

    generator.add("Dx_Q", Dx_Q["kiq"] <= G_Q["eiq"] * Dxi_Q["keq"])

    # [VE] K -> mu_eff
    generator.add(
        "assembleVolume",
        A["kl"]
        <= J_Q["q"] * W["q"] * MU_EFF["m"] * matE_Q_T["qm"] * Dx_Q["kiq"] * Dx_Q["liq"],
    )

    G_q = Tensor("G_q", (dim, dim, nq))
    n_q = Tensor("n_q", (dim, nq))
    n_unit_q = Tensor("n_unit_q", (dim, nq))
    nl_q = Tensor("nl_q", (nq,))
    w = Tensor("w", (nq,))
    E_q = [Tensor("E_q({})".format(x), (Nbf, nq)) for x in range(2)]
    matE_q_T = Tensor("matE_q_T", (nq, Nbf))
    Dxi_q = [Tensor("Dxi_q({})".format(x), (Nbf, dim, nq)) for x in range(2)]
    K_Dx_q = [Tensor("K_Dx_q({})".format(x), (Nbf, dim, nq)) for x in range(2)]
    a = [
        [Tensor("a({},{})".format(x, y), (Nbf, Nbf)) for y in range(2)]
        for x in range(2)
    ]
    c0 = [Scalar("c0{}".format(x)) for x in range(2)]
    c1 = [Scalar("c1{}".format(x)) for x in range(2)]
    c2 = [Scalar("c2{}".format(x)) for x in range(2)]
    Lift = [Tensor("Lift({})".format(x), (Nbf, dim, Nbf)) for x in range(2)]
    L_q = [Tensor("L_q({})".format(x), (Nbf, nq)) for x in range(2)]
    Minv = [Tensor("Minv({})".format(x), (Nbf, Nbf)) for x in range(2)]
    K_q = [Tensor("K_q({})".format(x), (nq,)) for x in range(2)]

    # [VE] K -> mu_eff.  prepare_penalty must also use mu_eff, not mu_0.
    generator.add(
        "K_Dx_q",
        K_Dx_q[0]["kiq"] <= MU_EFF["m"] * matE_q_T["qm"] * G_q["eiq"] * Dxi_q[0]["keq"],
    )
    generator.add("K_q", K_q[0]["q"] <= MU_EFF["m"] * matE_q_T["qm"])

    generator.addFamily(
        "lift_ip",
        simpleParameterSpace(2),
        lambda x: L_q[x]["lq"] <= E_q[x]["lq"] * nl_q["q"],
    )

    generator.addFamily(
        "lift_skeleton",
        simpleParameterSpace(2),
        lambda x: [
            Lift[0]["liu"]
            <= 0.5 * Minv[0]["us"] * E_q[0]["sq"] * E_q[x]["lq"] * n_q["iq"] * w["q"],
            Lift[1]["liv"]
            <= 0.5 * Minv[1]["vs"] * E_q[1]["sq"] * E_q[x]["lq"] * n_q["iq"] * w["q"],
            L_q[x]["lq"]
            <= 0.5
            * n_q["iq"]
            * (
                K_q[0]["q"] * E_q[0]["uq"] * Lift[0]["liu"]
                + K_q[1]["q"] * E_q[1]["vq"] * Lift[1]["liv"]
            ),
        ],
    )
    generator.add(
        "lift_boundary",
        [
            Lift[0]["liu"]
            <= Minv[0]["us"] * E_q[0]["sq"] * E_q[0]["lq"] * n_q["iq"] * w["q"],
            L_q[0]["lq"] <= n_q["iq"] * K_q[0]["q"] * E_q[0]["uq"] * Lift[0]["liu"],
        ],
    )

    def surface(x, y):
        return (
            a[x][y]["kl"]
            <= c0[y] * w["q"] * K_Dx_q[x]["kiq"] * n_q["iq"] * E_q[y]["lq"]
            + c1[x] * w["q"] * K_Dx_q[y]["liq"] * n_q["iq"] * E_q[x]["kq"]
            + c2[abs(y - x)] * w["q"] * E_q[x]["kq"] * L_q[y]["lq"]
        )

    generator.addFamily("assembleSurface", simpleParameterSpace(2, 2), surface)

    b = Tensor("b", (Nbf,), alignStride=petsc_alignment)
    F_Q = Tensor("F_Q", (Nq,))
    generator.add(
        "rhsVolume", b["k"] <= b["k"] + J_Q["q"] * W["q"] * E_Q["kq"] * F_Q["q"]
    )

    f_q = Tensor("f_q", (nq,))
    f_lifted = [Tensor("f_lifted({})".format(x), (Nbf, dim)) for x in range(2)]
    f_lifted_q = Tensor("f_lifted_q", (nq,))

    generator.add("rhs_lift_ip", f_lifted_q["q"] <= nl_q["q"] * f_q["q"])

    generator.add(
        "rhs_lift_boundary",
        [
            f_lifted[0]["li"]
            <= Minv[0]["lm"] * E_q[0]["mq"] * f_q["q"] * w["q"] * n_q["iq"],
            f_lifted_q["q"]
            <= n_q["iq"] * K_q[0]["q"] * E_q[0]["lq"] * f_lifted[0]["li"],
        ],
    )

    generator.add(
        "rhs_lift_skeleton",
        [
            f_lifted[0]["li"]
            <= 0.5 * Minv[0]["lm"] * E_q[0]["mq"] * f_q["q"] * w["q"] * n_q["iq"],
            f_lifted[1]["li"]
            <= 0.5 * Minv[1]["lm"] * E_q[1]["mq"] * f_q["q"] * w["q"] * n_q["iq"],
            f_lifted_q["q"]
            <= 0.5
            * n_q["iq"]
            * (
                K_q[0]["q"] * E_q[0]["lq"] * f_lifted[0]["li"]
                + K_q[1]["q"] * E_q[1]["lq"] * f_lifted[1]["li"]
            ),
        ],
    )

    generator.add(
        "rhsFacet",
        b["k"]
        <= b["k"]
        + c1[0] * w["q"] * K_Dx_q[0]["kiq"] * n_q["iq"] * f_q["q"]
        + c2[0] * w["q"] * E_q[0]["kq"] * f_lifted_q["q"],
    )

    # [VE] Viscoelastic history

    # RHS additions:   -integral_over_Element eps(v):sigmaHat^n  +  sum_F int_F [[v]]{sigmaHat^n n}
    #
    # ---- volume ----

    eps_n_Q = Tensor("eps_n_Q", (dim, Nq))  # eps^n at volume quad points
    eps_new_Q = Tensor("eps_new_Q", (dim, Nq))
    q_n_Q = Tensor("q_n_Q", (dim, Nq))
    q_new_Q = Tensor("q_new_Q", (dim, Nq))
    gdt_Q = Tensor("gdt_Q", (Nq,))
    ratio_Q = Tensor("ratio_Q", (Nq,))
    mu1_g_W_J_Q = Tensor("mu1_g_W_J_Q", (Nq,))
    mu1_r_W_J_Q = Tensor("mu1_r_W_J_Q", (Nq,))
    U_old = Tensor("U_old", (Nbf,), alignStride=petsc_alignment)

    # Precomputed per element, per time step (store next to AbsDetJWK).
    generator.add(
        "history_weights_Q",
        [
            gdt_Q["q"] <= GDT["m"] * matE_Q_T["qm"],
            ratio_Q["q"] <= RATIO["m"] * matE_Q_T["qm"],
            mu1_g_W_J_Q["q"]
            <= MU1["m"]
            * matE_Q_T["qm"]
            * GDT["n"]
            * matE_Q_T["qn"]
            * W["q"]
            * J_Q["q"],
            mu1_r_W_J_Q["q"]
            <= MU1["m"]
            * matE_Q_T["qm"]
            * RATIO["n"]
            * matE_Q_T["qn"]
            * W["q"]
            * J_Q["q"],
        ],
    )

    # eps at volume quad points from a modal solution vector.  Run Dx_Q first.
    generator.add("strain_Q", eps_n_Q["iq"] <= 0.5 * Dx_Q["kiq"] * U_old["k"])

    # -int eps(v):sigmaHat^n = +2 int mu_1 (g eps^n - r q^n) . grad v
    generator.add(
        "rhsVolumeHistory",
        b["k"]
        <= b["k"]
        + 2.0
        * (mu1_g_W_J_Q["q"] * eps_n_Q["jq"] - mu1_r_W_J_Q["q"] * q_n_Q["jq"])
        * Dx_Q["kjq"],
    )

    # End-of-step update.  q_new_Q and q_n_Q must be distinct buffers.
    generator.add(
        "update_history_Q",
        q_new_Q["iq"]
        <= ratio_Q["q"] * q_n_Q["iq"] + gdt_Q["q"] * (eps_new_Q["iq"] - eps_n_Q["iq"]),
    )

    # ---- facet ----
    #
    # Only side-0 kernels are generated; the host calls them twice with swapped
    # buffers, matching the K_q / K_Dx_q idiom.

    eps_n_q = [Tensor("eps_n_q({})".format(x), (dim, nq)) for x in range(2)]
    eps_new_q = [Tensor("eps_new_q({})".format(x), (dim, nq)) for x in range(2)]
    q_n_q = [Tensor("q_n_q({})".format(x), (dim, nq)) for x in range(2)]
    q_new_q = [Tensor("q_new_q({})".format(x), (dim, nq)) for x in range(2)]
    gdt_q = [Tensor("gdt_q({})".format(x), (nq,)) for x in range(2)]
    ratio_q = [Tensor("ratio_q({})".format(x), (nq,)) for x in range(2)]
    mu1_g_q = [Tensor("mu1_g_q({})".format(x), (nq,)) for x in range(2)]
    mu1_r_q = [Tensor("mu1_r_q({})".format(x), (nq,)) for x in range(2)]
    sigma_hist_q = [Tensor("sigma_hist_q({})".format(x), (dim, nq)) for x in range(2)]
    traction_hist_q = Tensor("traction_hist_q", (nq,))
    c3 = Scalar("c3")

    # Precomputed per facet side, per time step (store next to KJInv0/KJInv1).
    generator.add(
        "history_weights_q",
        [
            gdt_q[0]["q"] <= GDT["m"] * matE_q_T["qm"],
            ratio_q[0]["q"] <= RATIO["m"] * matE_q_T["qm"],
            mu1_g_q[0]["q"] <= MU1["m"] * matE_q_T["qm"] * GDT["n"] * matE_q_T["qn"],
            mu1_r_q[0]["q"] <= MU1["m"] * matE_q_T["qm"] * RATIO["n"] * matE_q_T["qn"],
        ],
    )

    generator.add(
        "strain_q",
        eps_n_q[0]["iq"] <= 0.5 * G_q["eiq"] * Dxi_q[0]["keq"] * U_old["k"],
    )

    generator.add(
        "update_history_q",
        q_new_q[0]["iq"]
        <= ratio_q[0]["q"] * q_n_q[0]["iq"]
        + gdt_q[0]["q"] * (eps_new_q[0]["iq"] - eps_n_q[0]["iq"]),
    )

    generator.add(
        "sigma_hist_q",
        sigma_hist_q[0]["iq"]
        <= 2.0
        * (mu1_r_q[0]["q"] * q_n_q[0]["iq"] - mu1_g_q[0]["q"] * eps_n_q[0]["iq"]),
    )

    # n_q carries the surface Jacobian, so w alone completes the measure --
    # same convention as assembleSurface / rhsFacet.
    generator.add(
        "tractionHistoryAverageSkeleton",
        traction_hist_q["q"]
        <= 0.5 * n_q["iq"] * (sigma_hist_q[0]["iq"] + sigma_hist_q[1]["iq"]),
    )
    generator.add(
        "tractionHistoryAverageBoundary",
        traction_hist_q["q"] <= n_q["iq"] * sigma_hist_q[0]["iq"],
    )

    # +int [[v]]{sigmaHat n}, [[v]] = v_0 - v_1 with n outward from element 0.
    # Host calls this twice per skeleton facet: b=B0, E_q(0)=E_q[localNo[0]],
    # c3=+1, then b=B1, E_q(0)=E_q[localNo[1]], c3=-1 -- exactly the rhsFacet
    # pattern in rhs_skeleton.  Skip entirely on Natural (Neumann) facets.
    generator.add(
        "rhsFacetHistory",
        b["k"] <= b["k"] + c3 * w["q"] * E_q[0]["kq"] * traction_hist_q["q"],
    )

    # ---- Neumann (traction) BC: int_Gamma t v ds ----
    # assemble_boundary already returns false for BC::Natural; these facets must
    # also be skipped by rhsFacet and rhsFacetHistory, since t is the TOTAL
    # prescribed traction and already contains the history contribution.
    traction_q = Tensor("traction_q", (nq,))
    generator.add(
        "rhsTraction",
        b["k"] <= b["k"] + w["q"] * E_q[0]["kq"] * traction_q["q"] * nl_q["q"],
    )

    # ==================================================================
    # matrix-free
    # ==================================================================

    U = Tensor("U", (Nbf,), alignStride=petsc_alignment)
    U_ext = Tensor("U_ext", (Nbf,), alignStride=petsc_alignment)
    U_new = Tensor("U_new", (Nbf,), alignStride=petsc_alignment)
    u_hat_q = Tensor("u_hat_q", (nq,))
    sigma_hat_q = Tensor("sigma_hat_q", (dim, nq))
    E_Q_T = Tensor("E_Q_T", (Nq, Nbf))
    negative_E_Q_T = Tensor("negative_E_Q_T", (Nq, Nbf))
    E_q_T = [Tensor("E_q_T({})".format(x), (nq, Nbf)) for x in range(2)]
    negative_E_q_T = [
        Tensor("negative_E_q_T({})".format(x), (nq, Nbf)) for x in range(2)
    ]
    Dxi_q_120 = [Tensor("Dxi_q_120({})".format(x), (dim, nq, Nbf)) for x in range(2)]
    J_W_K_Q = Tensor("J_W_K_Q", (Nq,))
    K_G_q = [Tensor("K_G_q({})".format(x), (dim, dim, nq)) for x in range(2)]

    # [VE] K -> mu_eff, so the matrix-free apply matches the assembled operator.
    generator.add(
        "J_W_K_Q", J_W_K_Q["q"] <= J_Q["q"] * W["q"] * MU_EFF["m"] * matE_Q_T["qm"]
    )
    generator.add("K_G_q", K_G_q[0]["eiq"] <= MU_EFF["m"] * matE_q_T["qm"] * G_q["eiq"])

    generator.add(
        "flux_u_skeleton",
        u_hat_q["q"]
        <= 0.5 * (negative_E_q_T[0]["ql"] * U["l"] + E_q_T[1]["ql"] * U_ext["l"]),
    )
    generator.add("flux_u_boundary", u_hat_q["q"] <= negative_E_q_T[0]["ql"] * U["l"])
    generator.add(
        "flux_sigma_skeleton",
        sigma_hat_q["pq"]
        <= 0.5
        * (
            K_G_q[0]["epq"] * Dxi_q_120[0]["eql"] * U["l"]
            + K_G_q[1]["epq"] * Dxi_q_120[1]["eql"] * U_ext["l"]
        )
        + c0[0]
        * (E_q_T[0]["ql"] * U["l"] + negative_E_q_T[1]["ql"] * U_ext["l"])
        * n_unit_q["pq"],
    )
    generator.add(
        "flux_sigma_boundary",
        sigma_hat_q["pq"]
        <= K_G_q[0]["epq"] * Dxi_q_120[0]["eql"] * U["l"]
        + c0[0] * E_q_T[0]["ql"] * U["l"] * n_unit_q["pq"],
    )
    generator.add(
        "apply_volume",
        [
            Dx_Q["krq"] <= Dxi_Q["keq"] * G_Q["erq"],
            U_new["k"] <= J_W_K_Q["q"] * Dx_Q["krq"] * Dx_Q["lrq"] * U["l"],
        ],
    )
    generator.add(
        "apply_facet",
        U_new["k"]
        <= U_new["k"]
        + w["q"]
        * n_q["rq"]
        * (
            u_hat_q["q"] * K_G_q[0]["erq"] * Dxi_q[0]["keq"]
            - E_q[0]["kq"] * sigma_hat_q["rq"]
        ),
    )

    # ==================================================================
    # traction (fault post-processing)
    # ==================================================================

    u = [
        Tensor("u({})".format(x), (Nbf,), alignStride=petsc_alignment) for x in range(2)
    ]
    grad_u = Tensor("grad_u", (dim, nq))
    generator.add(
        "grad_u",
        grad_u["pq"]
        <= 0.5 * (K_Dx_q[0]["lpq"] * u[0]["l"] + K_Dx_q[1]["lpq"] * u[1]["l"])
        + c0[0]
        * (E_q[0]["lq"] * u[0]["l"] - E_q[1]["lq"] * u[1]["l"] - f_q["q"])
        * n_unit_q["pq"],
    )

    generator.add(
        "grad_u_bnd",
        grad_u["pq"]
        <= K_Dx_q[0]["lpq"] * u[0]["l"]
        + c0[0] * (E_q[0]["lq"] * u[0]["l"] - f_q["q"]) * n_unit_q["pq"],
    )

    # [VE] grad_u carries only 2 mu_eff eps^{n+1}; the friction law needs the
    # full stress.  Call sigma_hist_q for the relevant sides first, then apply
    # these in traction_skeleton / traction_boundary.
    generator.add(
        "grad_u_history",
        grad_u["pq"]
        <= grad_u["pq"] + 0.5 * (sigma_hist_q[0]["pq"] + sigma_hist_q[1]["pq"]),
    )
    generator.add(
        "grad_u_history_bnd",
        grad_u["pq"] <= grad_u["pq"] + sigma_hist_q[0]["pq"],
    )
