mms5 = {}

mms5.t_e = 50.0
mms5.t_w = 1.0
mms5.rho0 = 2.670
mms5.cs = 3.464
mms5.nu = 0.25
mms5.E = 2.0 * mms5.cs^2 * mms5.rho0 * (1.0 + mms5.nu)
mms5.b = 0.02
mms5.V0 = 1.0e-6
mms5.f0 = 0.6
mms5.k = 3.14159265358979323846

function mms5:mu(x, y, z)
    return self.cs^2 * self.rho0
end

function mms5:lam(x, y, z)
    return 2 * self.nu * self:mu(x, y, z) / (1 - 2 * self.nu)
end

function mms5:eta(x, y, z)
    return self.cs * self.rho0 / 2.0
end

function mms5:L(x, y, z)
    return 0.2
end

function mms5:sn_pre(x, y, z)
    return 25 * math.abs(math.sin(self.k*z))
end

function mms5:tau_pre(x, y, z)
    return 20 * math.sin(self.k*z), 0.0
end

function mms5:a(x, y, z)
    return 0.015
end

function mms5:u_star(x, z, t)
    local nu = self.nu
    local E = self.E
    local k = self.k
    if x < 0 then
        k = -k
    end

    local K = self:K(t)
    local ux_star = (k*nu*x + k*x - 2*nu^2 - nu + 1)*K*math.exp(-k*x)*math.cos(k*z)/E
    local uz_star = (k*nu*x + k*x + 2*nu^2 - 2)*K*math.exp(-k*x)*math.sin(k*z)/E

    return ux_star, 0.0, uz_star
end

function mms5:boundary(x, y, z, t)
    return self:u_star(x, z, t)
end

function mms5:solution(x, y, z, t)
    return self:u_star(x, z, t)
end

function mms5:fault_solution(x, y, z, t)
    local S1 = self:S1_star(z, t)
    local psi = self:psi_star(z, t)
    return S1, 0.0, psi
end

function mms5:psi_star(z, t)
    local a = self:a(0, 0, z)
    local eta = self:eta(0, 0, z)
    local V1 = self:V1_star(z, t)
    local tau1_0, tau2_0 = self:tau_pre(0, 0, z)
    local tau1 = self:tau1_star(z, t) + tau1_0
    local sn = self:sn_star(z) - self:sn_pre(0, 0, z)
    local s = math.sinh((tau1 + eta * V1) / (a * sn))
    return a * math.log(2.0 * s * self.V0 / V1)
end

function mms5:dpsi_stardt(z, t)
    local a = self:a(0, 0, z)
    local eta = self:eta(0, 0, z)
    local sn = self:sn_star(z) - self:sn_pre(0, 0, z)
    local V1 = self:V1_star(z, t)
    local dV1dt = self:dV1_stardt(z, t)
    local tau1_0, tau2_0 = self:tau_pre(0, 0, z)
    local tau1 = self:tau1_star(z, t) + tau1_0
    local dtau1dt = self:dtau1_stardt(z, t)
    local th = math.tanh((eta * V1 + tau1) / (a * sn))
    return (-a * sn * dV1dt + eta * V1 / th * dV1dt + V1 * dtau1dt / th) / (sn * V1)
end

function mms5:tau1_star(z, t)
    return -self.k * self:K(t) * math.sin(self.k * z)
end

function mms5:dtau1_stardt(z, t)
    return -self.k * self:dKdt(t) * math.sin(self.k * z)
end

function mms5:sn_star(z)
    return 0.0
end

function mms5:S1_star(z, t)
    local factor = -4*(1-self.nu^2)/self.E
    return factor * self:K(t) * math.sin(self.k * z)
end

function mms5:V1_star(z, t)
    local factor = -4*(1-self.nu^2)/self.E
    return factor * self:dKdt(t) * math.sin(self.k * z)
end

function mms5:dV1_stardt(z, t)
    local factor = -4*(1-self.nu^2)/self.E
    return factor * self:d2Kdt2(t) * math.sin(self.k * z)
end

function mms5:K(t)
    local at = math.atan((t - self.t_e) / self.t_w)
    return 2.0 * (at + math.pi / 2.0) / math.pi
end

function mms5:dKdt(t)
    local ts = (t - self.t_e) / self.t_w
    return 2.0 / (math.pi * self.t_w * (1.0 + ts^2))
end

function mms5:d2Kdt2(t)
    return - 4.0 * self.t_w * (t - self.t_e) / (math.pi * (self.t_w^2 + (t - self.t_e)^2)^2)
end

function mms5:Vinit(x, y, z)
    return self:V1_star(z, 0), 0.0
end

function mms5:Sinit(x, y, z)
    return self:S1_star(z, 0), 0.0
end

function mms5:source(x, y, z, t)
    local ps = self:psi_star(z, t)
    local V1 = self:V1_star(z, t)
    local L = self:L(x, y, z)
    return - self.b * self.V0 / L * (math.exp((self.f0 - ps) / self.b) - math.abs(V1) / self.V0) + self:dpsi_stardt(z, t)
end
