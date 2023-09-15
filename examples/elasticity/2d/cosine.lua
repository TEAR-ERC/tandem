require "warp"

Cosine = {}

function Cosine:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function Cosine:warp(x, y)
    return partialAnnulus(x, y)
end

function Cosine:mu(x, y)
    return 1.0
end

function Cosine:lam(x, y)
    return 2.0
end

function Cosine:force(x, y)
    return 5.0*math.pi^2*math.cos(math.pi*x)*math.cos(math.pi*y),
          -3.0*math.pi^2*math.sin(math.pi*x)*math.sin(math.pi*y)
end

function Cosine:solution(x, y)
    return math.cos(math.pi * x) * math.cos(math.pi * y), 0
end

function Cosine:solution_jacobian(x, y)
    return -math.pi * math.sin(math.pi * x) * math.cos(math.pi * y),
           -math.pi * math.cos(math.pi * x) * math.sin(math.pi * y),
            0, 0
end

function Cosine:boundary(x, y)
    return self:solution(x, y)
end

cosine = Cosine:new()
