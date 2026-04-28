mms1 = {}

mms1.anno = 365.0 * 24.0 * 3600.0
mms1.t_e = 35.0 * mms1.anno
mms1.t_w = 10
mms1.V_p = 1e-9
mms1.V_min = 1e-12
mms1.delta = (mms1.V_p / 2.0 - mms1.V_min) * mms1.t_e
mms1.H = 8
mms1.tau_inf = 31.73
mms1.rho = 3.0
mms1.b = 0.02
mms1.V0 = 1.0e-6
mms1.f0 = 0.6

function mms1:mu(x, y)
    return 24
end

function mms1:eta(x, y)
    return math.sqrt(self:mu(x, y) * self.rho) / 2.0
end

function mms1:L(x, y)
    return 0.2
end

function mms1:sn_pre(s, y)
    return 50.0
end

function mms1:a(x, y)
    return 0.015
end

function mms1:u_star(x, y, t)
    local p = self:phi(x, y)
    return self.V_p * t * (1 - p) / 2.0  + self.delta * self:K(t) * p / 2.0 + self.tau_inf * x / self:mu(x, y)
end

function mms1:boundary(x, y, t)
    return self:u_star(x, y, t)
end

function mms1:solution(x, y, t)
    return self:u_star(x, y, t)
end

function mms1:psi_star(x, y, t)
    local axy = self:a(x, y)
    local Vs = self:V_star(x, y, t)
    local s = math.sinh((self:tau_star(x, y, t) - self:eta(x, y) * Vs) / (axy * self:sn_pre(x, y)))
    return axy * math.log(2.0 * s * self.V0 / Vs)
end

function mms1:dpsi_stardt(x, y, t)
    local axy = self:a(x, y)
    local e = self:eta(x, y)
    local sn = self:sn_pre(x, y)
    local Vs = self:V_star(x, y, t)
    local dVsdt = self:dV_stardt(x, y, t)
    local dtsdt = self:dtau_stardt(x, y, t)
    local th = math.tanh((e * Vs - self:tau_star(x, y, t)) / (axy * sn))
    return -(axy * sn * dVsdt + (Vs * dtsdt - e * Vs * dVsdt) / th) / (sn * Vs)
end

function mms1:tau_star(x, y, t)
    local dpdx = self:dphidx(x, y)
    return 0.5 * self:mu(x, y) * (self.delta * self:K(t) - self.V_p * t) * dpdx + self.tau_inf
end

function mms1:dtau_stardt(x, y, t)
    local dpdx = self:dphidx(x, y)
    return 0.5 * self:mu(x, y) * (self.delta * self:dKdt(t) - self.V_p) * dpdx
end

function mms1:V_star(x, y, t)
    local p = self:phi(x, y)
    return self.V_p * (1 - p) + self.delta * p * self:dKdt(t)
end

function mms1:dV_stardt(x, y, t)
    local p = self:phi(x, y)
    return self.delta * p * self:d2Kdt2(t)
end

function mms1:K(t)
    local at = math.atan((t - self.t_e) / self.t_w)
    return (at + math.pi / 2.0) / math.pi + self.V_min / self.delta * t
end

function mms1:dKdt(t)
    local ts = (t - self.t_e) / self.t_w
    return self.V_min / self.delta + 1.0 / (math.pi * self.t_w * (1.0 + ts^2))
end

function mms1:d2Kdt2(t)
    return - 2.0 * self.t_w * (t - self.t_e) / (math.pi * (self.t_w^2 + (t - self.t_e)^2)^2)
end

function mms1:phi(x, y)
    return self.H * (self.H + x) / ((self.H + x)^2 + y^2)
end

function mms1:dphidx(x, y)
    return self.H * (y^2 - (self.H + x)^2) / ((self.H + x)^2 + y^2)^2
end

function mms1:Vinit(x, y)
    return self:V_star(x, y, 0)
end

function mms1:Sinit(x, y)
    return 2.0 * self:u_star(x, y, 0)
end

function mms1:source(x, y, t)
    local ps = self:psi_star(x, y, t)
    local Vs = self:V_star(x, y, t)
    return - self.b * self.V0 / self:L(x, y) * (math.exp((self.f0 - ps) / self.b) - Vs / self.V0) + self:dpsi_stardt(x, y, t)
end
