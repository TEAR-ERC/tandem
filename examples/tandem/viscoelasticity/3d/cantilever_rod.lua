-- Adapted from Marques, S.P.C. & Creus, G.J., 2012. Computational Viscoelasticity, Springer
-- and Gharti et. al. https://academic.oup.com/gji/article/216/2/1364/5199199
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

function CantileverRod:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function CantileverRod:boundary(x, y, z, t)
    if x < 1e-9 then
        return 0.0, 0.0, 0.0
    end
end

function CantileverRod:traction_boundary(x, y, z, t)
    if t <= 6000 then
        if x > 1e-9 then -- Apply at free end (x = L)
            return 0.01, 0.0, 0.0
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

function CantileverRod:relaxationTime(x, y, z)
    return self.eta / self:mu1(x, y, z)
end

cantilever_rod = CantileverRod:new()
