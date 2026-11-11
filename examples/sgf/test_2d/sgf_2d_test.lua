-- Uniform elastic half plane. lam = mu gives nu = 0.25 (the surface
-- displacement of a 2D dislocation in a homogeneous half plane does not
-- depend on the moduli at all).
local Half = {}

function Half:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function Half:lam(x, y) return 1.0 end
function Half:mu(x, y) return 1.0 end

half = Half:new()
