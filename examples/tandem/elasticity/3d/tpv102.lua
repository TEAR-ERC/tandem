local TPV102 = {}
TPV102.__index = TPV102

TPV102.b = 0.012
TPV102.V0 = 1.0e-6
TPV102.f0 = 0.6
TPV102.Vini = 1e-12


TPV102.a0 = 0.008
TPV102.da = 0.008
TPV102.W = 15.0
TPV102.w = 3.0
TPV102.R = 3.0
TPV102.z0 = -7.5
TPV102.rho0 = 2.670
TPV102.cs = 3.464
TPV102.nu = 0.25

function TPV102.new()
    local self = setmetatable({}, TPV102)
    return self
end

function TPV102:boundary(x, y, z, t)
    return 0, 0, 0
end

function TPV102:rho(x, y, z)
    return self.rho0
end

function TPV102:mu(x, y, z)
    return self.cs^2 * self.rho0
end

function TPV102:lam(x, y, z)
    return 2 * self.nu * self:mu(x,y) / (1 - 2 * self.nu)
end

function TPV102:eta(x, y, z)
    return 0.0
end

function TPV102:L(x, y, z)
    return 0.02
end

function TPV102:Sinit(x, y, z)
    return 0.0, 0.0
end

function TPV102:Vinit(x, y, z)
    return 0.0, self.Vini
end

function TPV102:boxcar(x, W, w)
    s = math.abs(x)
    if s <= W then
        return 1.0
    elseif W < s and s < W + w then
        local t = w / (s - W - w) + w / (s - W)
        return 0.5 * (1.0 + math.tanh(t))
    else
        return 0.0
    end
end

function TPV102:a(x, y, z)
    local f = 1.0 - self:boxcar(x, self.W, self.w) * self:boxcar(z-self.z0, self.W/2.0, self.w)
    return self.a0 + self.da * f
end

function TPV102:sn_pre(x, y, z)
    return 120.0
end

function TPV102:tau_pre(x, y, z)
    return 0.0, -75.0
end

function TPV102:delta_tau(x, y, z, t)
    local r = math.sqrt(x^2 + (z-self.z0)^2)
    local F = 0.0
    if r < self.R then
        F = math.exp(r^2 / (r^2 - self.R^2))
    end
    local T = 1.0
    local G = 1.0
    if t < T then
        G = math.exp((t-T)^2 / (t*(t-2*T)))
    end
    return 0.0, -25.0 * F * G
end

tpv102 = TPV102.new()
