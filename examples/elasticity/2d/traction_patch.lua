-- Prescribed traction and traction-free boundary condition test on the unit
-- square [0,L] x [0,H].
--
-- Used by test/test_traction_patch_bc.py (2D).
--
-- left (tag 5):          Dirichlet, u = exact solution
-- right (tag 7):         prescribed traction t = sigma.n = (s, 0)
-- top, bottom (tag 1):   traction free (Natural), sigma.n = 0
--
-- s > 0 pulls the block (tension), s < 0 pushes it (compression). The exact
-- solution is a uniform uniaxial stress state in plane strain,
--
--   sigma_xx = s,   sigma_yy = sigma_xy = 0,
--   u_x = exx * x,   u_y = eyy * (y - H/2),
--   exx = s * (lam + 2 mu) / (4 mu (lam + mu)),   eyy = -lam / (lam + 2 mu) * exx.
--
-- The Dirichlet data on the left is the exact field, which contracts
-- laterally; clamping u = 0 there would break the uniaxial state. The top and
-- bottom carry the only traction-free (Natural) facets in the suite whose flux
-- an exact solution depends on: sigma.n = 0 holds there only if the solver
-- imposes no traction, so a wrong sign, a missing term or a spurious penalty
-- on a Natural facet lands in the error norm. Do not drop the tag-1 curves as
-- redundant.
--
-- traction_boundary returns the traction vector in global coordinates, i.e.
-- sigma.n with n the outward normal. On boundary facets tandem does not orient
-- it by ref_normal (DGCurvilinearCommon::make_facet_functional flips only on
-- interior facets), so the result is independent of ref_normal; the test
-- checks this by running with both signs.
--
-- As the solution is linear, the DG solution should match it up to solver
-- tolerance for any polynomial degree and mesh.
--
-- lam != mu so that mixing up the Lame parameters shows up in the error.

local L = 1.0
local H = 1.0

local TractionPatch = {}

function TractionPatch:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function TractionPatch:mu(x, y)
    return 1.0
end

function TractionPatch:lam(x, y)
    return 2.0
end

function TractionPatch:strains()
    local mu = self:mu(0.0, 0.0)
    local lam = self:lam(0.0, 0.0)
    local exx = self.s * (lam + 2.0 * mu) / (4.0 * mu * (lam + mu))
    local eyy = -lam / (lam + 2.0 * mu) * exx
    return exx, eyy
end

function TractionPatch:traction_boundary(x, y)
    return self.s, 0.0
end

function TractionPatch:solution(x, y)
    local exx, eyy = self:strains()
    return exx * x, eyy * (y - 0.5 * H)
end

function TractionPatch:boundary(x, y)
    return self:solution(x, y)
end

function TractionPatch:solution_jacobian(x, y)
    local exx, eyy = self:strains()
    return exx, 0.0, 0.0, eyy
end

traction_patch_tension = TractionPatch:new{s = 1.0}
traction_patch_compression = TractionPatch:new{s = -1.0}
