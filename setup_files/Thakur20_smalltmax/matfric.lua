local BP1 = {}

BP1.a_b1 = 0.012
BP1.a_b2 = -0.004
BP1.a_b3 = 0.015
BP1.a_b4 = 0.024
BP1.b = 0.015

BP1.tau1 = -10
BP1.tau2 = -30
BP1.tau3 = -22.5

BP1.Wf = 24
BP1.H = 12.0
BP1.h = 5.0
BP1.H2 = 2.0

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
    return 32
end

function BP1:eta(x, y)
    return math.sqrt(self:mu(x, y) * self.rho0) / 2.0
end

function BP1:L(x, y)
    return 0.004
end

function BP1:sn_pre(x, y)
    local z = -y
    local _sigma1 = 50.0 + (50.0 - 10.0) * (z - self.H2) / self.H2
    if z < self.H2 then
        return _sigma1
    else
        return 50.0
    end
end

function BP1:Vinit(x, y)
    return 1.0e-9
end

function BP1:ab(x, y)
    -- a_b = a_b4*np.ones(z.shape)
    -- a_b[z < Wf] = a_b3 + (a_b4-a_b3)*(z[z < Wf]-H-h)/(Wf-H-h)
    -- a_b[z < H+h] = a_b2 + (a_b3-a_b2)*(z[z < H+h]-H)/h
    -- a_b[z < H] =a_b2*np.ones(z[z < H].shape)
    -- a_b[z < H2] = a_b2 + (a_b2 - a_b1)*(z[z < H2]-H2)/H2
    local z = -y
    local _ab1 = self.a_b2 + (self.a_b2 - self.a_b1) * (z - self.H2) / self.H2
    local _ab2 = self.a_b2 + (self.a_b3 - self.a_b2) * (z - self.H) / self.h
    local _ab3 = self.a_b3 + (self.a_b4 - self.a_b3) * (z - self.h - self.H) / (self.Wf - self.h - self.H)

    if z < self.H2 then
        return _ab1
    elseif z < self.H then
        return self.a_b2
    elseif z < self.H + self.h then
        return _ab2
    elseif z < self.Wf then
        return _ab3
    else
        return self.a_b4
    end
end

function BP1:a(x, y)
    local z = -y
    local _ab = self:ab(x,y)
    return _ab + self.b
end

function BP1:tau_pre(x, y)
    -- local Vi = self:Vinit(x, y)
    -- local sn = self:sn_pre(x, y)
    -- local e = math.exp((self.f0 + self.b * math.log(self.V0 / Vi)) / self.amax)
    -- return -(sn * self.amax * math.asinh((Vi / (2.0 * self.V0)) * e) + self:eta(x, y) * Vi)
    local z = -y
    local _tau1 = self.tau2 + (self.tau2 - self.tau1) * (z - self.H2) / self.H2
    local _tau2 = self.tau2 + (self.tau3 - self.tau2) * (z - self.H) / self.h

    if z < self.H2 then
        return _tau1
    elseif z < self.H then
        return self.tau2
    elseif z < self.H + self.h then
        return _tau2
    else
        return self.tau3
    end
end

bp1 = BP1:new()

bp1_sym = BP1:new()
function bp1_sym:boundary(x, y, t)
    return self.Vp/2.0 * t
end

