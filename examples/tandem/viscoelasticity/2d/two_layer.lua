--[[
Creep of two standard linear solids stacked on top of each other, under the same
uniform shear stress.

Used by test/test_viscoelastic_two_layer.py. The constants are repeated in
viscoelastic_two_layer_config in test/conftest.py; change both together.

The unit square (two_layer.geo) is split at y = 0.5: the lower layer has relaxation
time tau = 1, the upper tau = 10, with the same mu0, mu1 and lam. The bottom is fixed,
and the sides and the top carry the traction of a uniform shear stress sigma_xy = s
from t = 0. Each layer then creeps on its own as an SLS in shear,

    gamma(t) = s/mu0 - (s/mu0 - s/(mu0 + mu1)) exp(-t / tau_c),   tau_c = tau (mu0 + mu1)/mu0,

and u_x(y) is the integral of gamma over the layers below y, u_y = 0.

With one relaxation time for the whole domain (the smallest), both layers would creep
at the rate of the lower one: at t = 10 the upper layer's strain would be 0.00997
instead of 0.00697, 43 % too large.

Scenarios:
  two_layer                  relaxation time derived as viscosity / mu1 (the normal use)
  two_layer_inconsistent     also gives relaxation_time, which disagrees with
                             viscosity / mu1 in the upper layer: must be rejected
]]--

local mu0 = 1.0
local mu1 = 1.0
local lam = 1.0
local s = 0.01          -- shear stress
local tau_low = 1.0
local tau_high = 10.0

local TwoLayer = {}
TwoLayer.__index = TwoLayer
-- the QD driver asks for these even with no fault
TwoLayer.f0 = 0
TwoLayer.V0 = 0
TwoLayer.b = 0
TwoLayer.theta = 0.1        -- dt cap = theta * smallest tau

function TwoLayer.new() return setmetatable({}, TwoLayer) end

local function tau(y)
    if y < 0.5 then return tau_low end
    return tau_high
end

function TwoLayer:mu0(x, y) return mu0 end
function TwoLayer:mu1(x, y) return mu1 end
function TwoLayer:lam(x, y) return lam end
function TwoLayer:viscosity(x, y) return mu1 * tau(y) end

function TwoLayer:boundary(x, y, t) return 0.0, 0.0 end

-- sigma.n of sigma_xy = s: top (n = +y) -> (s, 0), right (n = +x) -> (0, s),
-- left (n = -x) -> (0, -s)
function TwoLayer:traction_boundary(x, y, t)
    if y > 1.0 - 1e-9 then return s, 0.0 end
    if x > 1.0 - 1e-9 then return 0.0, s end
    return 0.0, -s
end

two_layer = TwoLayer.new()

-- Deprecated relaxation_time that disagrees with viscosity / mu1 above y = 0.5.
TwoLayerInconsistent = setmetatable({}, {__index = TwoLayer})
TwoLayerInconsistent.__index = TwoLayerInconsistent
function TwoLayerInconsistent.new() return setmetatable({}, TwoLayerInconsistent) end
function TwoLayerInconsistent:relaxation_time(x, y) return tau_low end

two_layer_inconsistent = TwoLayerInconsistent.new()
