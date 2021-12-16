local a = 0.2

local CircularHole = {}

function CircularHole:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function CircularHole:mu(x, y)
    return 1.0
end

function polar(x, y)
    local r = math.sqrt(x^2 + y^2)
    local t = math.atan2(y, x)
    return r, t
end

function CircularHole:solution(x, y)
    local r, t = polar(x, y)
    return (r^2 + a^4 / r^2) * math.sin(2*t)
end

function CircularHole:solution_jacobian(x, y)
    local r, t = polar(x, y)
    ux = -2*a^4*math.sin(3*t) / r^3 + 2*r*math.sin(t)
    uy =  2*a^4*math.cos(3*t) / r^3 + 2*r*math.cos(t)
    return ux, uy
end

function CircularHole:boundary(x, y)
    return self:solution(x, y)
end

circular_hole = CircularHole:new()
