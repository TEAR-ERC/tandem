-- Free slip boundary condition test on the unit cube [0,L] x [0,L] x [0,H].
--
-- Used by test/test_free_slip_bc.py (3D). 3D analogue of ../2d/free_slip.lua.
--
-- x = 0, x = L, y = 0, y = L: free slip with u.n = g (n = outward normal)
-- bottom (z = 0):             free slip with u.n = 0
-- top (z = H):                free surface
--
-- g > 0 stretches the block in x and y (extension), g < 0 squeezes it
-- (compression). The exact solution is homogeneous with zero shear stress:
--
--   u_x = eps * (x - L/2),   u_y = eps * (y - L/2),   eps = 2 * g / L
--   u_z = -2 * lam / (lam + 2 * mu) * eps * z
--
-- It satisfies u.n = g and zero tangential traction on the four sides, u_z = 0
-- and zero tangential traction at the bottom, and sigma.n = 0 at the top
-- (sigma_zz = lam * (2 eps + eps_zz) + 2 mu eps_zz = 0).
-- As the solution is linear, the DG solution should match it up to solver
-- tolerance for any polynomial degree and mesh.
--
-- lam != mu so that mixing up the Lame parameters shows up in the error.

local L = 1.0
local H = 1.0

local FreeSlip = {}

function FreeSlip:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function FreeSlip:mu(x, y, z)
    return 1.0
end

function FreeSlip:lam(x, y, z)
    return 2.0
end

function FreeSlip:strain()
    return 2.0 * self.g / L
end

function FreeSlip:vertical_strain()
    local mu = self:mu(0.0, 0.0, 0.0)
    local lam = self:lam(0.0, 0.0, 0.0)
    return -2.0 * lam / (lam + 2.0 * mu) * self:strain()
end

function FreeSlip:free_slip_boundary(x, y, z)
    if z < 1e-9 * H then
        return 0.0
    end
    return self.g
end

function FreeSlip:solution(x, y, z)
    local eps = self:strain()
    return eps * (x - 0.5 * L), eps * (y - 0.5 * L), self:vertical_strain() * z
end

function FreeSlip:solution_jacobian(x, y, z)
    local eps = self:strain()
    return eps, 0.0, 0.0,
           0.0, eps, 0.0,
           0.0, 0.0, self:vertical_strain()
end

free_slip_extension = FreeSlip:new{g = 1.0}
free_slip_compression = FreeSlip:new{g = -1.0}
