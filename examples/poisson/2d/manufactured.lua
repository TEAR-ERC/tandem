require "warp"

Manufactured = {}

function Manufactured:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function Manufactured:warp(x, y)
    return partialAnnulus(x, y)
end

function Manufactured:solution(x, y)
    return math.exp(-x - y^2)
end

function Manufactured:boundary(x, y)
    return self:solution(x, y)
end

function Manufactured:force(x, y)
    return (1.0 - 4.0 * y^2) * math.exp(-x - y^2)
end

manufactured = Manufactured:new()
