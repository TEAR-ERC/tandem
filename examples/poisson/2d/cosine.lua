require "warp"

local Cosine = {}

function Cosine:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function Cosine:warp(x, y)
    return partialAnnulus(x, y)
end

local f = 10.0

function cos1D(x)
    return math.cos(f * math.pi * x)
end

function Cosine:force(x, y)
    return 2 * f^2 * math.pi^2 * cos1D(x) * cos1D(y)
end

function Cosine:solution(x, y)
    return cos1D(x) * cos1D(y)
end

function Cosine:boundary(x, y)
    return self:solution(x, y)
end

cosine = Cosine:new()
