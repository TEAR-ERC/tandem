mms3 = {}

mms3.t_e = 50.0
mms3.t_w = 1.0
mms3.rho = 2.670
mms3.cs = 3.464
mms3.nu = 0.25
mms3.E = 2.0 * mms3.cs^2 * mms3.rho * (1.0 + mms3.nu)
mms3.b = 0.02
mms3.V0 = 1.0e-6
mms3.f0 = 0.6
mms3.k = 3.14159265358979323846

function mms3:mu(x, y)
    return self.cs^2 * self.rho
end

function mms3:lam(x, y)
    return 2 * self.nu * self:mu(x,y) / (1 - 2 * self.nu)
end

function mms3:eta(x, y)
    return self.cs * self.rho / 2.0
end

function mms3:L(x, y)
    return 0.2
end

function mms3:sn_pre(x, y)
    return 25 * math.abs(math.sin(self.k*y))
end

function mms3:tau_pre(x, y)
    return 20 * math.sin(self.k*y)
end

function mms3:a(x, y)
    return 0.015
end

function mms3:u_star(x, y, t)
    local nu = self.nu
    local E = self.E
    local k = self.k
    if x < 0 then
        k = -k
    end

    local K = self:K(t)
    local ux_star = (k*nu*x + k*x - 2*nu^2 - nu + 1)*K*math.exp(-k*x)*math.cos(k*y)/E
    local uy_star = (k*nu*x + k*x + 2*nu^2 - 2)*K*math.exp(-k*x)*math.sin(k*y)/E

    return ux_star, uy_star
end

function mms3:boundary(x, y, t)
    return self:u_star(x, y, t)
end

function mms3:solution(x, y, t)
    return self:u_star(x, y, t)
end

function mms3:fault_solution(x, y, t)
    local S = self:S_star(x, y, t)
    local psi = self:psi_star(x, y, t)
    return S, psi
end

function mms3:psi_star(x, y, t)
    local a = self:a(x, y)
    local Vs = self:V_star(x, y, t)
    local tau = self:tau_star(x, y, t) + self:tau_pre(x, y)
    local sn = self:sn_star(x, y) - self:sn_pre(x, y)
    local s = math.sinh((tau + self:eta(x, y) * Vs) / (a * sn))
    return a * math.log(2.0 * s * self.V0 / Vs)
end

function mms3:dpsi_stardt(x, y, t)
    local a = self:a(x, y)
    local e = self:eta(x, y)
    local sn = self:sn_star(x, y) - self:sn_pre(x, y)
    local Vs = self:V_star(x, y, t)
    local dVsdt = self:dV_stardt(x, y, t)
    local tau = self:tau_star(x, y, t) + self:tau_pre(x, y)
    local dtaudt = self:dtau_stardt(x, y, t)
    local th = math.tanh((e * Vs + tau) / (a * sn))
    return (-a * sn * dVsdt + e * Vs / th * dVsdt + Vs * dtaudt / th) / (sn * Vs)
end

function mms3:tau_star(x, y, t)
    return -self.k * self:K(t) * math.sin(self.k * y)
end

function mms3:dtau_stardt(x, y, t)
    return -self.k * self:dKdt(t) * math.sin(self.k * y)
end

function mms3:sn_star(x, y)
    return 0.0
end

function mms3:S_star(x, y, t)
    local factor = -4*(1-self.nu^2)/self.E
    return factor * self:K(t) * math.sin(self.k * y)
end

function mms3:V_star(x, y, t)
    local factor = -4*(1-self.nu^2)/self.E
    return factor * self:dKdt(t) * math.sin(self.k * y)
end

function mms3:dV_stardt(x, y, t)
    local factor = -4*(1-self.nu^2)/self.E
    return factor * self:d2Kdt2(t) * math.sin(self.k * y)
end

function mms3:K(t)
    local at = math.atan((t - self.t_e) / self.t_w)
    return 2.0 * (at + math.pi / 2.0) / math.pi
end

function mms3:dKdt(t)
    local ts = (t - self.t_e) / self.t_w
    return 2.0 / (math.pi * self.t_w * (1.0 + ts^2))
end

function mms3:d2Kdt2(t)
    return - 4.0 * self.t_w * (t - self.t_e) / (math.pi * (self.t_w^2 + (t - self.t_e)^2)^2)
end

function mms3:Vinit(x, y)
    return self:V_star(x, y, 0)
end

function mms3:Sinit(x, y)
    return self:S_star(x, y, 0)
end

function mms3:source(x, y, t)
    local ps = self:psi_star(x, y, t)
    local Vs = self:V_star(x, y, t)
    return - self.b * self.V0 / self:L(x, y) * (math.exp((self.f0 - ps) / self.b) - math.abs(Vs) / self.V0) + self:dpsi_stardt(x, y, t)
end
