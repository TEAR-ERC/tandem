#!/usr/bin/env python3


"""
YaTeTo kernel definitions for the Standard Linear Solid (Zener) model.

PHYSICS SUMMARY (from weak form derivation):

The total stress splits into a current-step part and a history part:

    σ_ij = σ̂ⁿ⁺¹_ij + σ̂ⁿ_ij

Where:
    σ̂ⁿ⁺¹_ij = Ĉ_ijkl(Δt) ε_kl               <- bilinear form (matrix A, assembled once per time step)
    σ̂ⁿ_ij   = 2μ₁ [exp(-Δt/τ) qⁿ_ij
                    - g(Δt) ε^{dev,n}_ij]   <- linear form  (RHS b, history source)

Effective stiffness tensor:
    Ĉ_ijkl(Δt) = A(Δt) δ_ij δ_kl + B(Δt)(δ_ik δ_jl + δ_il δ_jk)
    A(Δt) = λ + (2/3)(μ₁ - μ₁ g(Δt))      [≈ lam + 0.67*(mu1 - mu1*g)]
    B(Δt) = μ₀ + μ₁ g(Δt)

Time integration scalars (computed once per step in C++, passed as tensors):
    g(Δt)       = (τ/Δt)(1 - exp(-Δt/τ))   [closed form]
    exp(-Δt/τ)  = scalar exponential decay

Partial strain recurrence:
    qⁿ⁺¹_ij = qⁿ_ij exp(-Δt/τ) + g(Δt)(ε^{dev,n+1}_ij - ε^{dev,n}_ij)

Deviatoric strain:
    ε^{dev}_ij = ε_ij - (1/3) δ_ij ε_kk
              = (1/2)(∂u_i/∂x_j + ∂u_j/∂x_i) - (1/3) δ_ij (∂u_k/∂x_k)

========================================================================
WEAK FORM (SIPG, from derivation doc)
========================================================================

After splitting σ → σ̂ⁿ⁺¹ + σ̂ⁿ and multiplying by -1 for SPD, the full
discrete system is:

LHS (bilinear in u, v — becomes matrix A):
  + ∫_E   ε_ij(v) Ĉ_ijkl ε_kl(u) dE           [volume stiffness]
  - ∫_∂E  [[v_i]] {σ̂ⁿ⁺¹_ij(u) n_j} dS         [consistency]
  - ∫_ΓD∪ΓF [[u_i]] {σ̂ⁿ⁺¹_ij(v) n_j} dS       [symmetry]
  + γ ∫_∂E [[u_i]][[v_i]] dS                  [penalty]

RHS (linear in v — becomes vector b):
  + ∫_E   f_i v_i dE                          [body force]
  - ∫_ΓF  gᶠ_i {σ̂ⁿ⁺¹_ij(v) n_j} dS            [fault slip symmetry]
  - ∫_ΓD  gᴰ_i {σ̂ⁿ⁺¹_ij(v) n_j} dS            [Dirichlet symmetry]
  + γ ∫_ΓF  gᶠ_i [[v_i]] dS                   [fault slip penalty]
  + γ ∫_ΓD  gᴰ_i [[v_i]] dS                   [Dirichlet penalty]
  - ∫_E   ε_ij(v) σ̂ⁿ_ij dE                    [history volume]
  + ∫_∂E  [[v_i]] {σ̂ⁿ_ij n_j} dS              [history surface]

The first five RHS terms are identical to elasticity (handled by existing
rhs_boundary / rhs_volume kernels with effective params).

TENSOR INDEX CONVENTIONS:
  k, l    — basis function indices (0..Nbf-1)
  p, u    — displacement/force component indices (0..dim-1)
  i, j, r, s — free spatial indices (0..dim-1)
  Q, q    — volume / facet quadrature point indices
  [0],[1] — minus (interior) / plus (exterior) DG sides

"""

from . import stress, traction

from .tensors import make_tensors
from .helpers import make_helpers
from . import precompute
from . import assembly
from . import rhs
from . import strain


def add(generator, dim, nbf, Nbf, nq, Nq, petsc_alignment):
    """Add all viscoelasticity kernels with the YaTeTo generator.

    Args:
        generator (YaTeTo generator object): the generator to which kernels will be added
        dim (int): spatial dimension (2 or 3)
        nbf (int): number of facet basis functions
        Nbf (int): number of element basis functions
        nq (int): number of facet quadrature points
        Nq (int): number of volume quadrature points
        petsc_alignment (bool): Alignment flag for PETSc-aligned tensors
    """
    t = make_tensors(dim, Nbf, nq, Nq, petsc_alignment)
    h = make_helpers(t)
    # Attach helper closures to tensor namespace
    t.traction = h.traction
    t.tractionTest = h.tractionTest
    t.normalStressTest = h.normalStressTest
    t.test_normal = h.test_normal
    t.assembleTractionOp = h.assembleTractionOp
    t.tractionHistory = h.tractionHistory

    precompute.add(generator, t)
    assembly.add(generator, t)
    rhs.add(generator, t)
    strain.add(generator, t, dim)
    stress.add(generator, t)
    traction.add(generator, t)
