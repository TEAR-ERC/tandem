--[[
Simple shear of a standard linear solid across a fault that does not slip.

The unit box is cut from side to side by a horizontal fault at y = 0.5 and held
under Dirichlet control on all four sides with the displacement linear in y,
switched on at t = 0 and then held:

    antiplane (type poisson_viscoelasticity):  u   = ((y - H/2) * 2/H) * Umax
    inplane   (type viscoelasticity):          u_x = ((y - H/2) * 2/H) * Umax, u_y = 0

The friction parameters make the fault practically locked: with Vinit = 1e-32 m/s
and tau / (a sn) ~ 0.5 the slip stays below 1e-22 m, against Umax = 1e-3 m. The
displacement is then affine in y, which the mesh represents exactly, and the
shear traction on the fault is the uniform shear stress of the box,

    tau(t) = (mu0 + mu1 exp(-t / tau_relax)) * gamma,   gamma = 2 Umax / H,

with tau_relax = viscosity / mu1. The strain is constant after t = 0, which the
exponential integrator integrates exactly, so the written traction must match to
roundoff at every step, for any time step and Runge-Kutta scheme.

The test is aimed at the traction written by the monitor after a step. It must
combine the accepted displacement with the history of that step; with the
history already advanced to the end of the step it is off by
mu1 (1 - exp(-dt / tau_relax)) q, here 2.4 % at the start (theta = 0.05).

This file is used by test/test_viscoelastic_fault_traction.py. The material and
loading constants are repeated in viscoelastic_fault_traction_config in
test/conftest.py; change both together.
]]--

local mu0 = 30.0 -- Elastic shear modulus [GPa]
local mu1 = 30.0 -- Viscoelastic branch shear modulus [GPa]
local eta = 2.36682e10 -- Viscosity [GPa·s]
local nu = 0.25 -- Poisson's ratio
local Umax = 1.0e-3 -- Displacement at the top and bottom [m]
local H = 1.0 -- Domain height [km]

local FaultShear = {}
FaultShear.__index = FaultShear

FaultShear.f0 = 0.6
FaultShear.V0 = 1.0e-6
FaultShear.b = 0.015

-- Time step ratio (dt / tau_relax)
FaultShear.theta = 0.05

function FaultShear.new(inplane)
    local self = setmetatable({}, FaultShear)
    self.inplane = inplane
    return self
end

function FaultShear:mu0(x, y)
    return mu0
end

function FaultShear:mu1(x, y)
    return mu1
end

function FaultShear:viscosity(x, y)
    return eta
end

function FaultShear:lam(x, y)
    return 2.0 * mu0 * nu / (1.0 - 2.0 * nu)
end

function FaultShear:boundary(x, y, t)
    local u = ((y - 0.5 * H) * 2.0 / H) * Umax
    if self.inplane then
        return u, 0.0
    end
    return u
end

-- Friction: practically locked, see above. L is large so that the state
-- evolution stays far from the explicit stability limit at dt = theta * tau_relax.
function FaultShear:a(x, y)
    return 0.01
end

function FaultShear:eta(x, y)
    return 4.0
end

function FaultShear:L(x, y)
    return 1.0e4
end

function FaultShear:sn_pre(x, y)
    return 50.0
end

function FaultShear:tau_pre(x, y)
    return 0.0
end

function FaultShear:Vinit(x, y)
    return 1.0e-32
end

function FaultShear:Sinit(x, y)
    return 0.0
end

antiplane = FaultShear.new(false)
inplane = FaultShear.new(true)
