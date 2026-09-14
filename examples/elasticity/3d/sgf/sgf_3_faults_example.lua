-- Uniform elastic half space. Displacement from a prescribed-slip
-- dislocation depends only on Poisson's ratio, not on the absolute
-- stiffness, so lam = mu gives nu = 0.25 and cutde can be run with
-- nu = 0.25 for an exact comparison.
local TDE = {}

function TDE:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function TDE:lam(x, y, z) return 1.0 end
function TDE:mu(x, y, z) return 1.0 end

tde = TDE:new()
