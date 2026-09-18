-- Free slip boundary condition test on the unit square [0,L] x [0,H].
--
-- left/right: free slip with u.n = g (n = outward normal)
-- bottom:     free slip with u.n = 0
-- top:        free surface
--
-- g > 0 stretches the block (extension), g < 0 squeezes it (compression).
-- The exact solution is homogeneous plane strain with zero shear stress:
--
--   u_x = eps * (x - L/2),   eps = 2 * g / L
--   u_y = -lam / (lam + 2 * mu) * eps * y
--
-- It satisfies u.n = g and sigma_xy = 0 on the sides, u_y = 0 and
-- sigma_xy = 0 at the bottom, and sigma_yy = sigma_xy = 0 at the top.
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

function FreeSlip:mu(x, y)
    return 1.0
end

function FreeSlip:lam(x, y)
    return 2.0
end

function FreeSlip:strain()
    return 2.0 * self.g / L
end

function FreeSlip:free_slip_boundary(x, y)
    if y < 1e-9 * H then
        return 0.0
    end
    return self.g
end

function FreeSlip:solution(x, y)
    local eps = self:strain()
    local mu = self:mu(x, y)
    local lam = self:lam(x, y)
    return eps * (x - 0.5 * L), -lam / (lam + 2.0 * mu) * eps * y
end

function FreeSlip:solution_jacobian(x, y)
    local eps = self:strain()
    local mu = self:mu(x, y)
    local lam = self:lam(x, y)
    return eps, 0.0, 0.0, -lam / (lam + 2.0 * mu) * eps
end

free_slip_extension = FreeSlip:new{g = 1.0}
free_slip_compression = FreeSlip:new{g = -1.0}
