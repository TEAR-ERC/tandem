"""
Tensor namespace and memory layout definitions for viscoelasticity.

Defines the YaTeTo tensor objects, their dimensions, and memory alignments
used across all precomputation, assembly, and right-hand side kernels.
Variables are mapped to element quadrature nodes (Nq) or facet quadrature nodes (nq).
"""

from types import SimpleNamespace
import numpy as np
from yateto import Tensor, Scalar


def make_tensors(dim, Nbf, nq, Nq, petsc_alignment):
    t = SimpleNamespace()

    t.dim = dim
    # GEOMETRY AND QUADRATURE
    # Volume
    t.J = Tensor("J", (Nq,))
    t.Jinv_Q = Tensor("Jinv_Q", (Nq,))
    t.G = Tensor("G", (dim, dim, Nq))
    t.W = Tensor("W", (Nq,))

    # Facet
    t.g = [Tensor("g({})".format(x), (dim, dim, nq)) for x in range(2)]
    t.n_q = Tensor("n_q", (dim, nq))
    t.n_unit_q = Tensor("n_unit_q", (dim, nq))
    t.nl_q = Tensor("nl_q", (nq,))
    t.w = Tensor("w", (nq,))

    # MATERIAL PROPERTIES
    # Nodal values (Nbf) — interpolated to quad points via matE_Q_T/matE_q_T
    # in precompute kernels.
    t.lam = Tensor("lam", (Nbf,))
    t.mu0 = Tensor("mu0", (Nbf,))
    t.mu1 = Tensor("mu1", (Nbf,))
    t.viscosity = Tensor("viscosity", (Nbf,))
    t.rhoInv = Tensor("rhoInv", (Nbf,))

    # Interpolated to volume quad points (Nq)
    t.lam_Q = Tensor("lam_Q", (Nq,))
    t.mu0_Q = Tensor("mu0_Q", (Nq,))
    t.mu1_Q = Tensor("mu1_Q", (Nq,))
    t.viscosity_Q = Tensor("viscosity_Q", (Nq,))
    t.rhoInv_Q = Tensor("rhoInv_Q", (Nq,))

    # Precomputed combinations mat * W * J (Nq) — W*J folded in to avoid
    # repeating the multiply in every kernel.
    t.lam_W_J_Q = Tensor("lam_W_J_Q", (Nq,))
    t.mu0_W_J_Q = Tensor("mu0_W_J_Q", (Nq,))
    t.mu1_W_J_Q = Tensor("mu1_W_J_Q", (Nq,))
    t.viscosity_W_J_Q = Tensor("viscosity_W_J_Q", (Nq,))
    t.negative_rhoInv_W_Jinv_Q = Tensor("negative_rhoInv_W_Jinv_Q", (Nq,))

    # Effective Lamé params after time-step update, W*J-scaled (Nq)
    # A_dt = λ + (2/3) μ₁ (1 - g(Δt))
    # B_dt = μ₀ + μ₁ g(Δt)
    t.A_dt = Tensor("A_dt", (Nq,))
    t.B_dt = Tensor("B_dt", (Nq,))

    # Effective Lamé params, unscaled (Nq) — for stress output only
    t.A_dt_unscaled = Tensor("A_dt_unscaled", (Nq,))
    t.B_dt_unscaled = Tensor("B_dt_unscaled", (Nq,))

    # Interpolated to facet quad points (nq), two sides
    t.lam_q = [Tensor("lam_q({})".format(x), (nq,)) for x in range(2)]
    t.mu_q = [Tensor("mu_q({})".format(x), (nq,)) for x in range(2)]
    t.mu0_q = [Tensor("mu0_q({})".format(x), (nq,)) for x in range(2)]
    t.mu1_q = [Tensor("mu1_q({})".format(x), (nq,)) for x in range(2)]
    t.viscosity_q = [Tensor("viscosity_q({})".format(x), (nq,)) for x in range(2)]

    # Effective Lamé params at facet quad points (nq), two sides
    t.A_dt_q = [Tensor("A_dt_q({})".format(x), (nq,)) for x in range(2)]
    t.B_dt_q = [Tensor("B_dt_q({})".format(x), (nq,)) for x in range(2)]

    # ====================================================================
    # SECTION 3 — TIME INTEGRATION SCALARS
    # ====================================================================
    # Computed once per time step in C++, passed as (Nq,)/(nq,) tensors.
    # Spatially varying if τ differs per element.

    t.g_dt_Q = Tensor("g_dt_Q", (Nq,))  # g(Δt) = (τ/Δt)(1 - exp(-Δt/τ))
    t.g_dt_q = Tensor("g_dt_q", (nq,))
    t.ratio_Q = Tensor("ratio_Q", (Nq,))  # exp(-Δt/τ)
    t.ratio_q = Tensor("ratio_q", (nq,))

    # ====================================================================
    # SECTION 4 — BASIS FUNCTIONS
    # ====================================================================

    # Volume
    t.E_Q = Tensor("E_Q", (Nbf, Nq))
    t.matE_Q_T = Tensor("matE_Q_T", (Nq, Nbf))
    t.Dxi_Q = Tensor("Dxi_Q", (Nbf, dim, Nq))
    t.Dx_Q = Tensor("Dx_Q", (Nbf, dim, Nq))

    # Facet (two sides)
    t.E_q = [Tensor("E_q({})".format(x), (Nbf, nq)) for x in range(2)]
    t.matE_q_T = [Tensor("matE_q_T({})".format(x), (nq, Nbf)) for x in range(2)]
    t.Dxi_q = [Tensor("Dxi_q({})".format(x), (Nbf, dim, nq)) for x in range(2)]
    t.Dx_q = [Tensor("Dx_q({})".format(x), (Nbf, dim, nq)) for x in range(2)]

    # ====================================================================
    # SECTION 5 — DISPLACEMENT AND ASSEMBLY
    # ====================================================================

    t.U = Tensor("U", (Nbf, dim), alignStride=petsc_alignment)
    t.U_Q = Tensor("U_Q", (dim, Nq), alignStride=petsc_alignment)
    t.Unew = Tensor("Unew", (Nbf, dim), alignStride=petsc_alignment)

    t.M = Tensor("M", (Nbf, Nbf))
    t.matM = Tensor("matM", (Nbf, Nbf))
    t.A_mat = Tensor("A", (Nbf, dim, Nbf, dim))
    t.delta = Tensor("delta", (dim, dim), spp=np.identity(dim))

    # DG penalty scalars
    t.c0 = [Scalar("c0{}".format(x)) for x in range(2)]
    t.c1 = [Scalar("c1{}".format(x)) for x in range(2)]
    t.c2 = [Scalar("c2{}".format(x)) for x in range(2)]
    t.c3 = Scalar("c3")

    # Lifting
    t.Minv = [Tensor("Minv({})".format(x), (Nbf, Nbf)) for x in range(2)]
    t.Lift = [Tensor("Lift({})".format(x), (Nbf, dim, Nbf)) for x in range(2)]
    t.L_q = [Tensor("L_q({})".format(x), (Nbf, dim, dim, nq)) for x in range(2)]
    t.traction_op_q = [
        Tensor("traction_op_q({})".format(x), (Nbf, dim, dim, nq)) for x in range(2)
    ]

    t.u = [Tensor("u({})".format(x), (Nbf, dim)) for x in range(2)]
    t.a = [
        [Tensor("a({},{})".format(x, y), (Nbf, dim, Nbf, dim)) for y in range(2)]
        for x in range(2)
    ]

    # RHS working tensors
    t.traction_q = Tensor("traction_q", (dim, nq))
    t.f_q = Tensor("f_q", (dim, nq))
    t.f_lifted = [Tensor("f_lifted({})".format(x), (dim, Nbf, dim)) for x in range(2)]
    t.f_lifted_q = Tensor("f_lifted_q", (dim, nq))
    t.g_q = Tensor("g_q", (nq,))
    t.traction_component = Tensor("traction_component", (dim, nq))
    t.b = Tensor("b", (Nbf, dim), alignStride=petsc_alignment)
    t.b0 = Tensor("b0", (Nbf, dim), alignStride=petsc_alignment)
    t.b1 = Tensor("b1", (Nbf, dim), alignStride=petsc_alignment)
    t.F_Q = Tensor("F_Q", (dim, Nq))

    # STRAIN HISTORY
    # Layout: (quad_pt, dim, dim) throughout — matches C++ indexing.

    # Volume (Nq, dim, dim)
    t.strain_tensor_Q = Tensor("strain_tensor_Q", (Nq, dim, dim))
    t.strain_trace_Q = Tensor("strain_trace_Q", (Nq,))
    t.trace_tensor_Q = Tensor("trace_tensor_Q", (Nq, dim, dim))
    t.deviatoric_strain_tensor_Q = Tensor("deviatoric_strain_tensor_Q", (Nq, dim, dim))
    t.old_deviatoric_strain_tensor_Q = Tensor(
        "old_deviatoric_strain_tensor_Q", (Nq, dim, dim)
    )
    t.new_deviatoric_strain_tensor_Q = Tensor(
        "new_deviatoric_strain_tensor_Q", (Nq, dim, dim)
    )
    t.old_strain_tensor_Q = Tensor("old_strain_tensor_Q", (Nq, dim, dim))
    t.new_strain_tensor_Q = Tensor("new_strain_tensor_Q", (Nq, dim, dim))
    t.old_partial_strain_tensor_Q = Tensor(
        "old_partial_strain_tensor_Q", (Nq, dim, dim)
    )
    t.new_partial_strain_tensor_Q = Tensor(
        "new_partial_strain_tensor_Q", (Nq, dim, dim)
    )
    t.partial_strain_tensor_Q = Tensor("partial_strain_tensor_Q", (Nq, dim, dim))

    # Facet (nq, dim, dim), two sides
    t.strain_tensor_q = [
        Tensor("strain_tensor_q({})".format(x), (nq, dim, dim)) for x in range(2)
    ]
    t.strain_trace_q = [Tensor("strain_trace_q({})".format(x), (nq,)) for x in range(2)]
    t.trace_tensor_q = [
        Tensor("trace_tensor_q({})".format(x), (nq, dim, dim)) for x in range(2)
    ]
    t.deviatoric_strain_tensor_q = [
        Tensor("deviatoric_strain_tensor_q({})".format(x), (nq, dim, dim))
        for x in range(2)
    ]
    t.old_deviatoric_strain_tensor_q = [
        Tensor("old_deviatoric_strain_tensor_q({})".format(x), (nq, dim, dim))
        for x in range(2)
    ]
    t.new_deviatoric_strain_tensor_q = [
        Tensor("new_deviatoric_strain_tensor_q({})".format(x), (nq, dim, dim))
        for x in range(2)
    ]
    t.old_strain_tensor_q = [
        Tensor("old_strain_tensor_q({})".format(x), (nq, dim, dim)) for x in range(2)
    ]
    t.new_strain_tensor_q = [
        Tensor("new_strain_tensor_q({})".format(x), (nq, dim, dim)) for x in range(2)
    ]
    t.old_partial_strain_tensor_q = [
        Tensor("old_partial_strain_tensor_q({})".format(x), (nq, dim, dim))
        for x in range(2)
    ]
    t.new_partial_strain_tensor_q = [
        Tensor("new_partial_strain_tensor_q({})".format(x), (nq, dim, dim))
        for x in range(2)
    ]
    t.partial_strain_tensor_q = [
        Tensor("partial_strain_tensor_q({})".format(x), (nq, dim, dim))
        for x in range(2)
    ]

    # Precomputed history traction across facet
    t.tractionHistoryAverage = Tensor("tractionHistoryAverage", (nq, dim))
    t.tractionHistoryNormal = Tensor("tractionHistoryNormal", (nq,))

    # SECTION 7 — STRESS OUTPUT
    t.stress_total_Q = Tensor("stress_total_Q", (Nq, dim, dim))
    t.stress_nodal = Tensor("stress_nodal", (Nbf, dim, dim))

    return t
