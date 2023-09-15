local S = 1.0
local mu0 = 1.0
local nu = 0.25

Wedge = {}

function Wedge:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function Wedge:mu(x, y)
    return mu0
end

function Wedge:lam(x, y)
    return 2.0 * mu0 * nu / (1-2*nu)
end

function Wedge:solution(x, y)
    local r = math.sqrt(x^2 + y^2)
    local theta = math.atan2(y, x)

    local A_1 = (1.0/3.0)*math.sqrt(3)*S*mu0*(4*nu - 5)/(-16*nu^2 + 20*nu + 5)
    local A_2 = S*mu0*(7 - 4*nu)/(-16*nu^2 + 20*nu + 5)
    local A_3 = math.sqrt(3)*S*mu0*(5 - 4*nu)/(-16*nu^2 + 20*nu + 5)
    local A_4 = S*mu0*(4*nu - 7)/(-16*nu^2 + 20*nu + 5)
    local ur = (1.0/2.0)*r^2*(-3*A_1*math.sin(3*theta) - 3*A_2*math.cos(3*theta) - 4*A_3*nu*math.sin(theta) + A_3*math.sin(theta) - 4*A_4*nu*math.cos(theta) + A_4*math.cos(theta))/mu0
    local ut = (1.0/2.0)*r^2*(-3*A_1*math.cos(3*theta) + 3*A_2*math.sin(3*theta) - A_3*math.cos(theta) + A_4*math.sin(theta) + (4*nu - 4)*(A_3*math.cos(theta) - A_4*math.sin(theta)))/mu0

    local ux = ur * math.cos(theta) - ut * math.sin(theta)
    local uy = ur * math.sin(theta) + ut * math.cos(theta)
    return ux, uy
end

function Wedge:boundary(x, y)
    return self:solution(x, y)
end

wedge = Wedge:new()
