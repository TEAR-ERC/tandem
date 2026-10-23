--[[
Simple shear test using a Standard Linear Solid (SLS) rheology.

The material consists of an elastic spring (mu0) in parallel with a
Maxwell branch made of spring (mu1) and dashpot (eta). The Maxwell branch gives
a characteristic relaxation time

    tau_relax = eta / mu1

The loading is imposed as a Dirichlet displacement on all four sides of the
unit box, ramped linearly with y so the box is in simple shear:

    u_x(x, y, t) = ((y - H/2) * 2/H) * Umax * s(t)
    u_y          = 0

The engineering shear strain is therefore gamma(t) = (2*Umax/H) * s(t). The
scenarios below differ only in the scalar shape s(t):

    constant_strain_without_drop        s = 1
    constant_strain_with_drop           s = 1,            0 for t > T_hold
    constant_strain_rate_without_drop   s = t
    constant_strain_rate_with_drop      s = t,            0 for t > T_hold
    sinusoidal_without_drop             s = sin(omega t)
    sinusoidal_with_drop                s = sin(omega t), 0 for t > T_hold

`simple_shear` is the original example and is the same loading as
constant_strain_with_drop:
  - At loading: instantaneous elastic stress from both springs.
  - During hold: exponential stress relaxation toward the long-term elastic
    response controlled by mu0.
  - At unloading: instantaneous stress drop.
  - After release: transient reverse stress that decays exponentially to zero.

T_hold is chosen here as 6 relaxation times, so the viscoelastic branch is
nearly fully relaxed before unloading. The drop is sampled with `t <= T_hold`,
i.e. the load is still on at t = T_hold; the analytical reference in
test/conftest.py (viscoelastic_simple_shear_config) uses the same convention.

The constant strain rate cases reach gamma ~ 2e7 by the final time. They are a
time-integration exercise, not a physical loading.

This file is used by test/test_viscoelastic_simple_shear.py. The material
constants and the loading parameters are repeated in
viscoelastic_simple_shear_config in test/conftest.py; change both together.
]]--

local mu0 = 30.0 -- Elastic shear modulus [GPa]
local mu1 = 30.0 -- Viscoelastic branch shear modulus [GPa]
local eta = 2.36682e10 -- Viscosity [GPa·s]
local nu = 0.25 -- Poisson's ratio
local Umax = 1.0e-3 -- Maximum displacement magnitude at top [m]
local H = 1.0 -- Domain height [km]
local tau_relax = eta / mu1 -- Relaxation time [s]
local T_hold = 6.0 * tau_relax -- Hold duration before unloading [s]
local omega = 2.0 / tau_relax -- Sinusoid angular frequency [rad/s]

local SimpleShear = {}
SimpleShear.__index = SimpleShear

SimpleShear.f0 = 0

SimpleShear.V0 = 0
SimpleShear.b = 0

function SimpleShear.new(shape)
    local self = setmetatable({}, SimpleShear)
    self.shape = shape
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
    local ux = ((y - 0.5 * H) * 2.0 / H) * Umax * self.shape(t)
    local uy = 0.0
    return ux, uy
end

-- Time step ratio (dt / tau_relax)
SimpleShear.theta = 0.05

-- Imposed strain history shapes, s(t)
local function with_drop(shape)
    return function(t)
        if t <= T_hold then
            return shape(t)
        end
        return 0.0
    end
end

local function constant_strain(t)
    return 1.0
end

local function constant_strain_rate(t)
    return t
end

local function sinusoidal(t)
    return math.sin(omega * t)
end

constant_strain_without_drop = SimpleShear.new(constant_strain)
constant_strain_with_drop = SimpleShear.new(with_drop(constant_strain))
constant_strain_rate_without_drop = SimpleShear.new(constant_strain_rate)
constant_strain_rate_with_drop = SimpleShear.new(with_drop(constant_strain_rate))
sinusoidal_without_drop = SimpleShear.new(sinusoidal)
sinusoidal_with_drop = SimpleShear.new(with_drop(sinusoidal))

simple_shear = constant_strain_with_drop
