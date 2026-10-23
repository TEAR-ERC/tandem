--[[
Time-dependent viscoelastic MMS, run through the QD path with the new `force` hook.

Constitutive form actually assembled by the code (app/kernels/Viscoelasticity/__init__.py):
    sigma = Chat(dt) eps  +  2 mu1 [ exp(-dt/tau) q^n - g(dt) eps^{dev,n} ]
    A(dt) = lam + (2/3)(mu1 - mu1 g),   B(dt) = mu0 + mu1 g
    g(dt) = (tau/dt)(1 - exp(-dt/tau))        <- NOTE: not 1 - exp(-dt/tau)
    q^{n+1} = exp(-dt/tau) q^n + g(dt)(eps^{dev,n+1} - eps^{dev,n})

The manufactured field is DIVERGENCE FREE, u = curl(psi z), psi = sin(pi x) sin(pi y):
    U1 =  pi sin(pi x) cos(pi y),   U2 = -pi cos(pi x) sin(pi y),   div U = 0
so tr(eps) = 0 and the whole volumetric branch drops out. That removes any ambiguity
about how lam and the bulk modulus map onto each other, and leaves a pure test of the
deviatoric/memory path. psi is not harmonic, so f is nonzero and the history is
genuinely exercised (unlike a harmonic u, which needs no body force at all).

    u(x,t) = (1 - e^{-a t}) U(x),   eps^dev = (1 - e^{-a t}) E,
    E = pi^2 cos(pi x) cos(pi y) diag(1, -1)   (traceless)

The solution STARTS FROM REST. That is not cosmetic. The code carries eps^{dev,n}
and starts it at zero, so a field that is nonzero at t = 0 is seen as a jump from 0
to eps(0) and the first recursion step injects q = g(dt) eps(0), with g ~ 1. A
manufactured q(0) = 0 then disagrees with the code by O(eps(0)) at every later time.
An earlier version of this file used u = e^{-a t} U and produced an L2 error of 0.315
at t = 0 that never converged under mesh refinement - that was this mismatch, not
discretisation. With u(0) = 0 the two agree: the t = 0 solve gives u = 0, hence
eps^{dev,0} = 0 and q^0 = 0.

Continuous memory law dq/dt + q/tau = d eps^dev/dt with q(0) = 0 gives
    q(t) = Aq E (e^{-a t} - e^{-t/tau}),   Aq = a tau / (1 - a tau)
so the exact stress is sigma = M(t) E with
    M(t) = 2 mu0 (1 - e^{-a t}) + 2 mu1 Aq (e^{-a t} - e^{-t/tau})
and the body force f = -div sigma is
    f1 =  M(t) pi^3 sin(pi x) cos(pi y)
    f2 = -M(t) pi^3 cos(pi x) sin(pi y)

With mu0 = 2, mu1 = 1.5, tau = 0.5, a = 1:  Aq = 1 and
    M(t) = 4(1 - e^{-t}) + 3(e^{-t} - e^{-2t}),
M(0) = 0 (at rest) and M -> 2 mu0 = 4 as t -> infinity (fully relaxed).

Two scenarios share the material and the time history above:

  mms       the trigonometric field above. It is not in the discrete space at any
            degree, so the L2 error at t = 1 measures the spatial order (p + 1) once
            theta is small enough for the time error to drop out.

  mms_poly  the same construction with the cubic stream function
            psi = x^2 y + x y^2, i.e.
                U1 = x^2 + 2 x y,   U2 = -(2 x y + y^2),   div U = 0,
                E11 = -E22 = 2 (x + y),   E12 = E21 = x - y        (traceless),
                f  = -M(t) div E = M(t) (-1, 1).
            For p >= 2 the discrete space contains U, so the spatial error of the
            computed field only multiplies the (small) time error: the L2 error
            at t = 1 is the time-integration error alone and shows the temporal
            order (2) with no spatial floor, on a coarse mesh.

Used by test/test_viscoelastic_mms.py; the constants are repeated in
viscoelastic_mms_config in test/conftest.py.
]]--

local pi = math.pi

local mu0 = 2.0
local mu1 = 1.5
local tau = 0.5
local a   = 1.0

local Aq = a * tau / (1.0 - a * tau)           -- = +1 for a*tau = 0.5

local function M(t)
    return 2.0 * mu0 * (1.0 - math.exp(-a * t))
         + 2.0 * mu1 * Aq * (math.exp(-a * t) - math.exp(-t / tau))
end

MMS = {}
MMS.__index = MMS

-- the QD driver asks for these even with no fault
MMS.f0 = 0
MMS.V0 = 0
MMS.b  = 0
MMS.theta = 0.01                                -- dt cap = theta * tau

function MMS.new() return setmetatable({}, MMS) end

function MMS:mu0(x, y)             return mu0 end
function MMS:mu1(x, y)             return mu1 end
function MMS:relaxation_time(x, y) return tau end
function MMS:viscosity(x, y)       return mu1 * tau end
-- inactive: tr(eps) = 0 for this field, but the operator still wants a value
function MMS:lam(x, y)             return 1.0 end

function MMS:solution(x, y, t)
    local d = 1.0 - math.exp(-a * t)
    return  d * pi * math.sin(pi * x) * math.cos(pi * y),
           -d * pi * math.cos(pi * x) * math.sin(pi * y)
end

function MMS:boundary(x, y, t)
    return self:solution(x, y, t)
end

function MMS:force(x, y, t)
    local m = M(t)
    return  m * pi^3 * math.sin(pi * x) * math.cos(pi * y),
           -m * pi^3 * math.cos(pi * x) * math.sin(pi * y)
end

mms = MMS.new()

-- Polynomial variant, see the header. Inherits the material and theta from MMS.
MMSPoly = setmetatable({}, {__index = MMS})
MMSPoly.__index = MMSPoly

function MMSPoly.new() return setmetatable({}, MMSPoly) end

function MMSPoly:solution(x, y, t)
    local d = 1.0 - math.exp(-a * t)
    return  d * (x * x + 2.0 * x * y),
           -d * (2.0 * x * y + y * y)
end

function MMSPoly:force(x, y, t)
    local m = M(t)
    return -m, m
end

mms_poly = MMSPoly.new()
