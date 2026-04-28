--[[
Simple shear relaxation test using a Standard Linear Solid (SLS) rheology.

The material consists of an elastic spring (mu0) in parallel with a
Maxwell branch made of spring (mu1) and dashpot (eta). The Maxwell branch gives
a characteristic relaxation time

    tau_relax = eta / mu1

A box-strain loading is imposed through the boundary condition:
  - For 0 <= t <= T_hold:
        a constant simple shear displacement is applied, ramped linearly with y,
        reaching ±Umax at the top/bottom boundaries.
  - For t > T_hold:
        the imposed displacement is removed instantaneously (strain returns to 0).

Expected response:
  - At loading: instantaneous elastic stress from both springs.
  - During hold: exponential stress relaxation toward the long-term elastic
    response controlled by mu0.
  - At unloading: instantaneous stress drop.
  - After release: transient reverse stress that decays exponentially to zero.

T_hold is chosen here as 6 relaxation times, so the viscoelastic branch is
nearly fully relaxed before unloading.
]]--

-- Simple shear relaxation test
local mu0 = 30.0 -- Elastic shear modulus [GPa]
local mu1 = 30.0 -- Viscoelastic branch shear modulus [GPa]
local eta = 2.36682e10 -- Viscosity [GPa·s]
local nu = 0.25 -- Poisson's ratio
local Umax = 1.0e-3 -- Maximum displacement magnitude at top [m]
local H = 1.0 -- Domain height [km]
local tau_relax = eta / mu1 -- Relaxation time [s]
local T_hold = 6.0 * tau_relax -- Hold duration before unloading [s]

local mu = 30.0
local SimpleShear = {}
SimpleShear.__index = SimpleShear

SimpleShear.f0 = 0

SimpleShear.V0 = 0
SimpleShear.b = 0

function SimpleShear.new()
    local self = setmetatable({}, SimpleShear)
    return self
end

function SimpleShear:mu0(x, y)
    return mu0
end

-- Shear modulus (viscoelastic branch)
function SimpleShear:mu1(x, y)
    return mu1
end

function SimpleShear:viscosity(x, y)
    return eta
end

function SimpleShear:relaxation_time(x, y)
    return self:viscosity(x, y) / self:mu1(x, y)
end

function SimpleShear:lam(x, y)
    return 2.0 * mu0 * nu / (1.0 - 2.0 * nu)
end

function SimpleShear:boundary(x, y, t)
    local ux = 0.0
    local uy = 0.0
    if (t <= T_hold) then
        ux = ((y - 0.5 * H) * 2.0 / H) * Umax
    else
        ux = 0.0
    end
    return ux, uy

end

-- Time step ratio (dt / tau_relax)
SimpleShear.theta = 0.1
simple_shear = SimpleShear.new()
