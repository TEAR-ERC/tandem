local BP1 = {}

BP1.a0 = 0.010
BP1.b0 = 0.015
BP1.amax = 0.025
BP1.H = 15.0
BP1.h = 3.0
BP1.Vp = 1e-9
BP1.rho0 = 2.670
BP1.V0 = 1.0e-6
BP1.f0 = 0.6

function BP1:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function BP1:boundary(x, y, t, r)
    if x > 1.0 then
        return self.Vp/2.0 * t
    elseif x < -1.0 then
        return -self.Vp/2.0 * t
    else
        return self.Vp * t
    end
end

function BP1:mu(x, y, r)
    return 32.038120320
end

function BP1:eta(x, y, r)
    return math.sqrt(self:mu(x, y, r) * self.rho0) / 2.0
end

function BP1:L(x, y, r)
    return 0.008
end

function BP1:sn_pre(x, y, r)
    print(x,y)
    return 50.0
end

function BP1:Vinit(x, y, r)
    return 1.0e-9
end

function BP1:a(x, y, r)
    local z = -y
    if z < self.H then
        return self.a0
    elseif z < self.H + self.h then
        return self.a0 + (self.amax - self.a0) * (z - self.H) / self.h
    else
        return self.amax
    end
end

function BP1:b(x, y, r)
    return self.b0
end

function BP1:tau_pre(x, y, r)
    local Vi = self:Vinit(x, y)
    local sn = self:sn_pre(x, y)
	local b  = self:b(x, y)
    local e = math.exp((self.f0 + b * math.log(self.V0 / Vi)) / self.amax)
    return -(sn * self.amax * math.asinh((Vi / (2.0 * self.V0)) * e) + self:eta(x, y) * Vi)
end

bp1 = BP1:new()

bp1_sym = BP1:new()
function bp1_sym:boundary(x, y, t, r)
    return self.Vp/2.0 * t
end

