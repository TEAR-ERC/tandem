a0 = 0.004
amax = 0.04
h_s = 2.0
h_t = 2.0
H = 12.0
l = 60
Vp = 1e-9
Vzero = 1e-20
rho = 2.670
cs = 3.464
nu = 0.25

function boundary(x, y, z, t)
    local Vh = Vp * t
    if y > 1 then
        Vh = Vh / 2.0
    elseif y < -1 then
        Vh = -Vh / 2.0
    end
    return Vh, 0, 0
end

function mu(x, y, z)
    return cs^2 * rho
end

function lam(x, y, z)
    return 2 * nu * mu(x,y) / (1 - 2 * nu)
end

function eta(x, y, z)
    return cs * rho / 2.0
end

function Sinit(x, y, z)
    return 0.0, -0.0
end

function Vinit(x, y, z)
    return -Vzero, -Vp
end

function a(x, y, z)
    local d = -z
    local s = math.abs(x)
    if h_s + h_t <= d and d <= h_s + h_t + H and s <= l/2 then
        return a0
    elseif d <= h_s or h_s + 2*h_t + H <= d or l/2 + h_t <= s then
        return amax
    else
        local r = math.max(math.abs(d-h_s-h_t-H/2)-H/2, s-l/2)/h_t
        return a0 + r*(amax-a0)
    end
end

function sn_pre(x, y, z)
    return 25.0
end

function tau_pre(x, y, z)
    local f0 = 0.6
    local b = 0.03
    local V0 = 1e-6
    local Vi1, Vi2 = Vinit(x, y, z)
    local Vi = math.sqrt(Vi1^2 + Vi2^2)
    local sn = sn_pre(x, y, z)
    local e = math.exp((f0 + b * math.log(V0 / Vp)) / amax)
    local tau0 = sn * amax * math.asinh((Vp / (2.0 * V0)) * e) + eta(x, y, z) * Vp
    return tau0 * Vi1 / Vi, tau0 * Vi2 / Vi
end
