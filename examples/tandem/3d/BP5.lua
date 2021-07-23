local BP5 = {}
BP5.__index = BP5

BP5.b = 0.03
BP5.V0 = 1.0e-6
BP5.f0 = 0.6
BP5.Vp = 1e-9
BP5.Vzero = 1e-20

BP5.a0 = 0.004
BP5.amax = 0.04
BP5.h_s = 2.0
BP5.h_t = 2.0
BP5.H = 12.0
BP5.w = 12.0
BP5.l = 60
BP5.rho = 2.670
BP5.cs = 3.464
BP5.nu = 0.25

function BP5.new(params)
    local self = setmetatable({}, BP5)
    self.eps = params.eps
    return self
end

function BP5:boundary(x, y, z, t)
    local Vh = self.Vp * t
    if y > 1 then
        Vh = Vh / 2.0
    elseif y < -1 then
        Vh = -Vh / 2.0
    end
    return Vh, 0, 0
end

function BP5:mu(x, y, z)
    return self.cs^2 * self.rho
end

function BP5:lam(x, y, z)
    return 2 * self.nu * self:mu(x,y) / (1 - 2 * self.nu)
end

function BP5:eta(x, y, z)
    return self.cs * self.rho / 2.0
end

function BP5:in_nucleation(x, y, z)
    local d = -z
    local s = x
    local eps = self.eps
    if self.h_s + self.h_t <= d+eps and d-eps <= self.h_s + self.h_t + self.H and -self.l/2.0 <= s+eps and s-eps <= -self.l/2.0 + self.w then
        return true
    end
    return false
end

function BP5:L(x, y, z)
    if self:in_nucleation(x, y, z) then
        return 0.13
    end
    return 0.14
end

function BP5:Sinit(x, y, z)
    return 0.0, 0.0
end

function BP5:Vinit(x, y, z)
    if self:in_nucleation(x, y, z) then
        return self.Vzero, 0.01
    end
    return self.Vzero, self.Vp
end

function BP5:a(x, y, z)
    local d = -z
    local s = math.abs(x)
    if self.h_s + self.h_t <= d and d <= self.h_s + self.h_t + self.H and s <= self.l/2 then
        return self.a0
    elseif d <= self.h_s or self.h_s + 2*self.h_t + self.H <= d or self.l/2 + self.h_t <= s then
        return self.amax
    else
        local r = math.max(math.abs(d-self.h_s-self.h_t-self.H/2)-self.H/2, s-self.l/2)/self.h_t
        return self.a0 + r*(self.amax-self.a0)
    end
end

function BP5:sn_pre(x, y, z)
    return 25.0
end

function BP5:tau_pre(x, y, z)
    local Vi1, Vi2 = self:Vinit(x, y, z)
    local Vi = math.sqrt(Vi1^2 + Vi2^2)
    local sn = self:sn_pre(x, y, z)
    local ax = self:a(x, y, z)
    local e = math.exp((self.f0 + self.b * math.log(self.V0 / self.Vp)) / ax)
    local tau0 = sn * ax * math.asinh((Vi2 / (2.0 * self.V0)) * e)
                 + self:eta(x, y, z) * Vi2
    return -tau0 * Vi1 / Vi, -tau0 * Vi2 / Vi
end

return BP5
