local CS = {}

CS.a0 = 0.010
CS.b0 = 0.015
CS.H = 15.0
CS.h = 3.0
CS.Vp = 1e-9
CS.rho0 = 2.670
CS.V0 = 1.0e-6
CS.f0 = 0.6
CS.mu0 = 32.038120320

function CS:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function CS:boundary(x, y, t)
	return self.Vp * t
end

function CS:mu(x, y, r)
	print("CS:mu(x, y, r) -> ", r)
	return self.mu0
end

function CS:eta(x, y)
	return math.sqrt(self.mu0 * self.rho0) / 2.0
end

function CS:L(x, y)
	return 0.008
end

function CS:sn_pre(x, y)
	return 50.0
end

function CS:Vinit(x, y)
	return 1.0e-9
end

function CS:a(x, y)
	return self.a0
end

function CS:b(x, y)
	return self.b0
end

function CS:tau_pre(x, y)
	local Vi = self:Vinit(x, y)
    local sn = self:sn_pre(x, y)
	local b  = self:b(x, y)
    local e = math.exp((self.f0 + b * math.log(self.V0 / Vi)) / self.a0)
    return -(sn * self.a0 * math.asinh((Vi / (2.0 * self.V0)) * e) + self:eta(x, y) * Vi)
end

cs = CS:new()
