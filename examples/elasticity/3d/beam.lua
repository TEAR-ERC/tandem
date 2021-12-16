local Beam = {}

function Beam:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function Beam:force(x, y, z)
    local rho = 1.0
    local W = 0.2
    local L = 1.0
    local delta = W / L 
    local gamma = 0.4 * delta^2
    return 0.0, 0.0, -rho * gamma
end

function Beam:boundary(x, y, z)
    return 0.0, 0.0, 0.0
end

function Beam:lam(x, y, z)
    return 1.25
end

function Beam:mu(x, y, z)
    return 1.0
end

beam = Beam:new()
