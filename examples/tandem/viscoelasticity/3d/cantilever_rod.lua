-- Adapted from Marques, S.P.C. & Creus, G.J., 2012. Computational Viscoelasticity, Springer
-- and Gharti et. al. https://academic.oup.com/gji/article/216/2/1364/5199199
--
-- Used by test/test_viscoelastic_cantilever.py; the material constants and the load
-- history are repeated in viscoelastic_cantilever_config in test/conftest.py.
local CantileverRod = {}
CantileverRod.E = 1.0 -- GPa (1000 MPa instantaneous)
CantileverRod.nu = 0.4833
CantileverRod.r1 = 0.901 -- Prony series coefficient
CantileverRod.eta = 30.3714690218 -- viscosity in GPa·s (303.707 MPa × 98.99 s / 1000)
CantileverRod.L = 1e-4 -- 100 mm in km
CantileverRod.dy = 3e-5 -- 30 mm in km
CantileverRod.dz = 3e-5 -- 30 mm in km
CantileverRod.T_load = 6000.0 -- seconds (as per Gharti/ Abaqus example)
CantileverRod.Txx = 0.01 -- GPa = 10 MPa
CantileverRod.theta = 0.1

-- the QD driver asks for these even with no fault
CantileverRod.f0 = 0
CantileverRod.V0 = 0
CantileverRod.b = 0

function CantileverRod:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

-- Clamped end (tag 5, clamped = 1 in cantilever_rod.geo)
function CantileverRod:boundary(x, y, z, t)
    if x < 1e-9 then
        return 0.0, 0.0, 0.0
    end
end

-- Symmetry planes x = 0, y = 0, z = 0 (tag 9, clamped = 0 in cantilever_rod.geo)
function CantileverRod:free_slip_boundary(x, y, z, t)
    return 0.0
end

-- The load is on at t = T_load and off after it.
function CantileverRod:traction_boundary(x, y, z, t)
    if t <= self.T_load then
        if x > 1e-9 then -- Apply at free end (x = L)
            return self.Txx, 0.0, 0.0
        end
    else
        return 0.0, 0.0, 0.0
    end
end

-- not really used in tandem - here as a reference
function CantileverRod:mu_total()
    return 0.337086226656778804 -- in GPa (337.078 MPa)
end

function CantileverRod:mu0(x, y, z)
    return (1 - 0.901) * 0.337086226656778804 -- in GPa (337.078 MPa)
end

function CantileverRod:mu1(x, y, z)
    return (0.901) * 0.337086226656778804 -- in GPa (337.078 MPa)
end

function CantileverRod:lam(x, y, z)

    return 10 - (2.0 / 3.0) * 0.337086226656778804 -- in GPa (337.078 MPa)
end

function CantileverRod:viscosity(x, y, z)
    return self.eta
end

-- Must be spelled relaxation_time: a misspelt member is silently replaced by 1 s.
function CantileverRod:relaxation_time(x, y, z)
    return self.eta / self:mu1(x, y, z)
end

cantilever_rod = CantileverRod:new()
