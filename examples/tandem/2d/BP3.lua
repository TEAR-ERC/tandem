local BP3 = {}
BP3.__index = BP3

BP3.a0 = 0.010
BP3.amax = 0.025
BP3.H = 15.0
BP3.h = 3.0
BP3.rho = 2.670
BP3.cs = 3.464
BP3.nu = 0.25

function BP3.new(dip, Vp)
    local self = setmetatable({}, BP3)
    self.dip = dip * math.pi / 180.0
    self.Vp = Vp
    return self
end

function BP3:boundary(x, y, t)
    local Vh = self.Vp * t
    local dist = x + y / math.tan(self.dip)
    if dist > 1 then
        Vh = -Vh / 2.0
    elseif dist < -1 then
        Vh = Vh / 2.0
    end
    return Vh * math.cos(self.dip), -Vh * math.sin(self.dip)
end

function BP3:mu(x, y)
    return self.cs^2 * self.rho
end

function BP3:lam(x, y)
    return 2 * self.nu * self:mu(x,y) / (1 - 2 * self.nu)
end

function BP3:eta(x, y)
    return self.cs * self.rho / 2.0
end

function BP3:Sinit(x, y)
    return 0.0
end

function BP3:Vinit(x, y)
    return self.Vp
end

function BP3:a(x, y)
    local d = math.abs(y) / math.sin(self.dip)
    if d < self.H then
        return self.a0
    elseif d < self.H + self.h then
        return self.a0 + (self.amax - self.a0) * (d - self.H) / self.h
    else
        return self.amax
    end
end

function BP3:sn_pre(x, y)
    -- positive in compression
    return 50.0
end

function BP3:tau_pre(x, y)
    local f0 = 0.6
    local b = 0.015
    local V0 = 1e-6
    local Vi = self:Vinit(x, y)
    local sn = self:sn_pre(x, y)
    local e = math.exp((f0 + b * math.log(V0 / math.abs(Vi))) / self.amax)
    return -(sn * self.amax * math.asinh((Vi / (2.0 * V0)) * e) + self:eta(x, y) * Vi)
end

return BP3
