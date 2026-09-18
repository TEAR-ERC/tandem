-- poisson_volumeTagging.lua

local kA = 1.0
local kB = 2.0

local phi0 = 0.0
local phi1 = 1.0

local x_volumeTagging = 0.0

local volumeTagging = {}

function volumeTagging:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

-- Conductivity "K" for Poisson scenario
function volumeTagging:mu(x, y, tag)
    if tag == 1 then
        return kA
    else
        return kB
    end
end

-- analytical value of field at interface
local phi_star = (kB * phi1 + kA * phi0) / (kA + kB)

-- analytical solution
function volumeTagging:solution(x, y)
    if x <= x_volumeTagging then
        -- left side [-1,0]
        return phi0 + (phi_star - phi0) * (x + 1.0) / 1.0
    else
        -- right side [0,1]
        return phi_star + (phi1 - phi_star) * (x - 0.0) / 1.0
    end
end

-- Dirichlet BC on left/right
function volumeTagging:boundary(x, y)
    local bc = self:solution(x, y)
    return bc

end

volume_tagging = volumeTagging:new()
