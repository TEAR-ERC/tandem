anno = 365.0 * 24.0 * 3600.0
t_e = 35.0 * anno
t_w = 10
V_p = 1e-9
V_min = 1e-12
delta = (V_p / 2.0 - V_min) * t_e
H = 8
tau_inf = 31.73
rho = 3.0
b = 0.02
V0 = 1.0e-6
L = 0.2
f0 = 0.6
sn = 50

function mu(x, y)
    return 24
end

function eta(x, y)
    return math.sqrt(mu(x, y) * rho) / 2.0
end

function a(x, y)
    return 0.015
end

function u_star(x, y, t)
    local p = phi(x, y)
    return V_p * t * (1 - p) / 2.0  + delta * K(t) * p / 2.0 + tau_inf * x / mu(x, y)
end

function psi_star(x, y, t)
    local axy = a(x, y)
    local Vs = V_star(x, y, t)
    local s = sinh((tau_star(x, y, t) - eta(x, y) * Vs) / (axy * sn))
    return axy * math.log(2.0 * s * V0 / Vs)
end

function dpsi_stardt(x, y, t)
    local axy = a(x, y)
    local e = eta(x, y)
    local Vs = V_star(x, y, t)
    local dVsdt = dV_stardt(x, y, t)
    local dtsdt = dtau_stardt(x, y, t)
    local th = tanh((e * Vs - tau_star(x, y, t)) / (axy * sn))
    return - (axy * sn * dVsdt + (Vs * dtsdt - e * Vs * dVsdt) / th) / (sn * Vs)
end

function tau_star(x, y, t)
    local dpdx = dphidx(x, y)
    return 0.5 * mu(x, y) * (delta * K(t) - V_p * t) * dpdx + tau_inf
end

function dtau_stardt(x, y, t)
    local dpdx = dphidx(x, y)
    return 0.5 * mu(x, y) * (delta * dKdt(t) - V_p) * dpdx
end

function V_star(x, y, t)
    local p = phi(x, y)
    return V_p * (1 - p) + delta * p * dKdt(t)
end

function dV_stardt(x, y, t)
    local p = phi(x, y)
    return delta * p * d2Kdt2(t)
end

function K(t)
    local at = math.atan((t - t_e) / t_w)
    return (at + math.pi / 2.0) / math.pi + V_min / delta * t
end

function dKdt(t)
    local ts = (t - t_e) / t_w
    return V_min / delta + 1.0 / (math.pi * t_w * (1.0 + ts^2))
end

function d2Kdt2(t)
    return - 2.0 * t_w * (t - t_e) / (math.pi * (t_w^2 + (t - t_e)^2)^2)
end

function phi(x, y)
    return H * (H + x) / ((H + x)^2 + y^2)
end

function dphidx(x, y)
    return H * (y^2 - (H + x)^2) / ((H + x)^2 + y^2)^2
end

function sinh(x)
    return 0.5 * (math.exp(x) - math.exp(-x))
end

function cosh(x)
    return 0.5 * (math.exp(x) + math.exp(-x))
end

function tanh(x)
    return sinh(x) / cosh(x)
end

function Vinit(x, y)
    return V_star(x, y, 0)
end

function Sinit(x, y)
    return 2.0 * u_star(x, y, 0)
end

function source(x, y, t)
    local ps = psi_star(x, y, t)
    local Vs = V_star(x, y, t)
    return - b * V0 / L * (math.exp((f0 - ps) / b) - Vs / V0) + dpsi_stardt(x, y, t)
end
