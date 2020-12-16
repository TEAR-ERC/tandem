a0 = 0.010
amax = 0.025
H = 15.0
h = 3.0
Vp = 1e-9
rho = 2.670
cs = 3.464
nu = 0.25
dip = 60.0 * math.pi / 180.0
--dip = 90.0 * math.pi / 180.0

function boundary(x, y, t)
    local Vh = -Vp * t
    local dist = x + y / math.tan(dip)
    if dist > 1 then
        Vh = -Vh / 2.0
    elseif dist < -1 then
        Vh = Vh / 2.0
    end
    return Vh * math.cos(dip), -Vh * math.sin(dip)
end

function mu(x, y)
    return cs^2 * rho
end

function lam(x, y)
    return 2 * nu * mu(x,y) / (1 - 2 * nu)
end

function eta(x, y)
    return cs * rho / 2.0
end

function Sinit(x, y)
    return 0.0
end

function Vinit(x, y)
    return Vp
end

function a(x, y)
    local d = math.abs(y) / math.sin(dip)
    if d < H then
        return a0
    elseif d < H + h then
        return a0 + (amax - a0) * (d - H) / h
    else
        return amax
    end
end

function sn_pre(x, y)
    return 50.0
end

function tau_pre(x, y)
    local f0 = 0.6
    local b = 0.015
    local V0 = 1e-6
    local Vi = Vinit(x, y)
    local sn = sn_pre(x, y)
    local e = math.exp((f0 + b * math.log(V0 / math.abs(Vi))) / amax)
    return sn * amax * math.asinh((Vi / (2.0 * V0)) * e) + eta(x, y) * Vi
end
