local BP1 = {}

BP1.a0 = 0.02
BP1.amax = 0.023
BP1.H = 9.0
BP1.h = 6.0
BP1.H2 = 2.0
BP1.Wf = 24
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

function BP1:boundary(x, y, t)
    if x > 1.0 then
        return self.Vp/2.0 * t
    elseif x < -1.0 then
        return -self.Vp/2.0 * t
    else
        return self.Vp * t
    end
end

function BP1:mu(x, y)
    return 32.038120320
end

function BP1:eta(x, y)
    return math.sqrt(self:mu(x, y) * self.rho0) / 2.0
end

function BP1:L(x, y)
    return 0.004
end

function BP1:sn_pre(x, y)
    return 50.0
end

function BP1:Vinit(x, y)
    return 1.0e-9
end

function BP1:a(x, y)
    return 0.02
end

function BP1:ab(x, y)
    return 0.005
end

function BP1:b(x, y)
    return 0.015
end

function BP1:tau_pre(x, y)
    return -30.0
end

bp1 = BP1:new()

bp1_sym = BP1:new()
function bp1_sym:boundary(x, y, t)
    return self.Vp/2.0 * t
end

