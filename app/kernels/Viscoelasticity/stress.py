"""
Stress computation and projection kernels for viscoelasticity.

Generates the YaTeTo kernels for computing the macroscopic
viscoelastic stress tensor at the quadrature points. It also includes the L2
projection RHS kernel used to map the quadrature-point stresses back onto the
nodal basis for visualization (e.g., Paraview/VTK output).
"""


def add(generator, t):
    generator.add(
        "computeTotalStress",
        t.stress_total_Q["Qij"]
        <= (
            t.A_dt_unscaled["Q"] * t.delta["ij"] * t.strain_trace_Q["Q"]
            + 2 * t.B_dt_unscaled["Q"] * t.strain_tensor_Q["Qij"]
            + 2.0
            * t.mu1_Q["Q"]
            * (
                t.ratio_Q["Q"] * t.old_partial_strain_tensor_Q["Qij"]
                - t.g_dt_Q["Q"] * t.old_deviatoric_strain_tensor_Q["Qij"]
            )
        ),
    )

    generator.add(
        "projectStressRHS",
        t.stress_nodal["kij"]
        <= t.matE_Q_T["Qk"] * t.W["Q"] * t.J["Q"] * t.stress_total_Q["Qij"],
    )
