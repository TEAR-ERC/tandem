"""
Strain and internal variable history kernels for viscoelasticity.

Generates the YaTeTo kernels for computing the kinematic
quantities (strain, trace, deviatoric strain) from the displacement field.
It also defines the recurrence relations for the viscoelastic internal
variables (partial strains) required by the Standard Linear Solid (Zener) model
at both volume and facet quadrature points.
"""

from yateto import simpleParameterSpace


def add(generator, t, dim):
    generator.add(
        "strain_tensor_Q",
        # ε_ij(u) = (1/2)(∂u_i/∂x_j + ∂u_j/∂x_i)
        # Layout: strain_tensor_Q["Qij"] where Q=quad pt, i,j=spatial
        [
            t.strain_tensor_Q["Qij"]
            <= 0.5 * (t.Dx_Q["bjQ"] * t.U["bi"] + t.Dx_Q["biQ"] * t.U["bj"]),
        ],
    )

    generator.add(
        "strain_trace_Q",
        # Tr(ε) = ε_kk = ∂u_k/∂x_k (volumetric strain)
        [t.strain_trace_Q["Q"] <= t.strain_tensor_Q["Qij"] * t.delta["ji"]],
    )

    generator.add(
        "trace_tensor_Q",
        # Tr(ε) δ_ij  (isotropic part of strain, to be subtracted for deviatoric)
        [t.trace_tensor_Q["Qij"] <= t.strain_trace_Q["Q"] * t.delta["ij"]],
    )

    generator.add(
        "deviatoric_strain_tensor_Q",
        # ε^{dev}_ij = ε_ij - (1/D) Tr(ε) δ_ij
        # where D = dim. Note: 1/3 in 3D, 1/2 in 2D (from kernel param dim).
        [
            t.deviatoric_strain_tensor_Q["Qij"]
            <= t.strain_tensor_Q["Qij"] + (-1.0 / dim) * t.trace_tensor_Q["Qij"],
        ],
    )

    generator.add(
        "partialStrain_Q",
        # Recurrence for partial strain (Maxwell element internal variable):
        # qⁿ⁺¹_ij = qⁿ_ij * ratio_Q + g_dt_Q * (ε^{dev,n+1}_ij - ε^{dev,n}_ij)
        # ratio_Q  = exp(-Δt/τ) at each volume quad pt
        # g_dt_Q   = g(Δt) at each volume quad pt
        [
            t.new_partial_strain_tensor_Q["Qij"]
            <= (
                t.old_partial_strain_tensor_Q["Qij"] * t.ratio_Q["Q"]
                + t.g_dt_Q["Q"]
                * (
                    t.new_deviatoric_strain_tensor_Q["Qij"]
                    - t.old_deviatoric_strain_tensor_Q["Qij"]
                )
            )
        ],
    )

    generator.add(
        "updateStrain_Q",
        # Rotate buffers: old ← new  (called after partialStrain_Q)
        [t.old_strain_tensor_Q["Qij"] <= t.new_strain_tensor_Q["Qij"]],
    )

    # FACET STRAIN COMPUTATION KERNELS
    # Same as Section 12 but at facet quadrature points.
    # Needed because the history surface term ∫ [[v]] {σ̂ⁿ n} dS requires
    # strain values at the facet, not the interior.

    generator.addFamily(
        "strain_tensor_q",
        simpleParameterSpace(2),
        lambda x: [
            t.strain_tensor_q[x]["Qij"]
            <= 0.5 * (t.Dx_q[x]["bjQ"] * t.U["bi"] + t.Dx_q[x]["biQ"] * t.U["bj"]),
        ],
    )

    generator.addFamily(
        "strain_trace_q",
        simpleParameterSpace(2),
        lambda x: [
            t.strain_trace_q[x]["Q"] <= t.strain_tensor_q[x]["Qij"] * t.delta["ji"],
        ],
    )

    generator.addFamily(
        "trace_tensor_q",
        simpleParameterSpace(2),
        lambda x: [
            t.trace_tensor_q[x]["Qij"] <= t.strain_trace_q[x]["Q"] * t.delta["ij"],
        ],
    )

    generator.addFamily(
        "deviatoric_strain_tensor_q",
        simpleParameterSpace(2),
        lambda x: [
            t.deviatoric_strain_tensor_q[x]["Qij"]
            <= t.strain_tensor_q[x]["Qij"] + (-1.0 / dim) * t.trace_tensor_q[x]["Qij"],
        ],
    )

    generator.addFamily(
        "partialStrain_q",
        simpleParameterSpace(2),
        # Same recurrence as volume but at facet quad pts.
        # ratio_q = exp(-Δt/τ) at facet pts, g_dt_q = g(Δt) at facet pts.
        lambda x: [
            t.new_partial_strain_tensor_q[x]["qij"]
            <= (
                t.old_partial_strain_tensor_q[x]["qij"] * t.ratio_q["q"]
                + t.g_dt_q["q"]
                * (
                    t.new_deviatoric_strain_tensor_q[x]["qij"]
                    - t.old_deviatoric_strain_tensor_q[x]["qij"]
                )
            )
        ],
    )

    generator.addFamily(
        "updateStrain_q",
        simpleParameterSpace(2),
        lambda x: [t.old_strain_tensor_q[x]["qij"] <= t.new_strain_tensor_q[x]["qij"]],
    )
