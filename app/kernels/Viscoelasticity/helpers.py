"""
Symbolic formulae for viscoelastic tensor contractions.

Provides helper functions that bind to the tensor namespace to generate 
reusable YaTeTo kernels for traction, lifting, and history operators. 
These functions do not execute math; they return symbolic expressions 
for the code generator.
"""

from types import SimpleNamespace


def make_helpers(t):
    """
    Build and return all helper closures, bound to the tensor namespace t.
    Returns a SimpleNamespace so callers can do h.traction(x, normal) etc.
    """

    def traction(x, normal):
        """
        Traction vector σ̂ⁿ⁺¹_pj(u^x) normal_j at facet quad pts.

        σ̂ⁿ⁺¹_pj n_j = A_dt (∂u_s/∂x_s) n_p
                      + B_dt (∂u_p/∂x_j n_j + ∂u_j/∂x_j n_p)

        Uses effective params A_dt_q (≈ λ_eff) and B_dt_q (≈ μ_eff).
        Indices: l=basis, p=component, q=quad pt, j,s=spatial free.
        """
        return t.A_dt_q[x]["q"] * t.Dx_q[x]["lsq"] * t.u[x]["ls"] * normal[
            "pq"
        ] + t.B_dt_q[x]["q"] * (
            t.Dx_q[x]["ljq"] * t.u[x]["lp"] * normal["jq"]
            + t.Dx_q[x]["lpq"] * t.u[x]["lj"] * normal["jq"]
        )

    def tractionTest(x, utilde):
        """
        Symmetry traction σ̂ⁿ⁺¹_kj(v) n_j applied to test function v.

        utilde["iq"] is the test function value at quad pts (or a BC value).
        Indices: k=basis row, p=component, q=quad pt.
        """
        return t.A_dt_q[x]["q"] * t.Dx_q[x]["kpq"] * utilde["iq"] * t.n_q[
            "iq"
        ] + t.B_dt_q[x]["q"] * (
            t.Dx_q[x]["kjq"] * utilde["pq"] * t.n_q["jq"]
            + t.Dx_q[x]["kiq"] * utilde["iq"] * t.n_q["pq"]
        )

    def normalStressTest(x):
        """
        Normal stress operator σ̂ⁿ⁺¹_nn applied to test function v:
          A_dt ∂v_k/∂x_p + 2 B_dt n̂_p (∂v_k/∂x_j n̂_j)

        Used for free-slip BC where only the normal component is penalised.
        Indices: k=basis, p=component, q=quad pt.
        """
        return t.A_dt_q[x]["q"] * t.Dx_q[x]["kpq"] + 2.0 * t.B_dt_q[x][
            "q"
        ] * t.n_unit_q["pq"] * (t.Dx_q[x]["kjq"] * t.n_unit_q["jq"])

    def test_normal(x):
        """
        C_ijus n_j for the lift / BR2 operator:
          A_dt δ_us n_i + B_dt (δ_iu n_s + δ_is n_u)

        Indices: i,u,s spatial, q=quad pt.
        """
        return t.A_dt_q[x]["q"] * t.delta["us"] * t.n_q["iq"] + t.B_dt_q[x]["q"] * (
            t.delta["iu"] * t.n_q["sq"] + t.delta["is"] * t.n_q["uq"]
        )

    def assembleTractionOp(x):
        """
        Traction operator T^x_{kpuq} = σ̂ⁿ⁺¹_{pu}(φ_k) n_u for one side x.

        Stored into traction_op_q[x]["kpuq"]:
          A_dt ∂φ_k/∂x_p n_u + B_dt (n_j ∂φ_k/∂x_j δ_{pu} + ∂φ_k/∂x_u n_p)
        """
        return t.traction_op_q[x]["kpuq"] <= (
            t.A_dt_q[x]["q"] * t.Dx_q[x]["kpq"] * t.n_q["uq"]
            + t.B_dt_q[x]["q"]
            * (
                t.n_q["jq"] * t.Dx_q[x]["kjq"] * t.delta["pu"]
                + t.Dx_q[x]["kuq"] * t.n_q["pq"]
            )
        )

    def tractionHistory(side, normal):
        """
        History traction at facet quad pts for one side:
          μ₁ exp(-Δt/τ) qⁿ_pj n_j  −  μ₁ g(Δt) ε^{dev,n}_pj n_j

        The supplied normal controls whether this is evaluated with n_q
        (surface-form terms) or n_unit_q (physical traction output).

        The factor of 2 from σ̂ⁿ = 2μ₁[...] is applied by the calling kernel.

        Layout: partial_strain_tensor_q[side]["qpj"] and deviatoric["qpj"].
        """
        return (
            t.mu1_q[side]["q"]
            * t.ratio_q["q"]
            * t.partial_strain_tensor_q[side]["qpj"]
            * normal["jq"]
            - t.mu1_q[side]["q"]
            * t.g_dt_q["q"]
            * t.deviatoric_strain_tensor_q[side]["qpj"]
            * normal["jq"]
        )

    return SimpleNamespace(
        traction=traction,
        tractionTest=tractionTest,
        normalStressTest=normalStressTest,
        test_normal=test_normal,
        assembleTractionOp=assembleTractionOp,
        tractionHistory=tractionHistory,
    )
