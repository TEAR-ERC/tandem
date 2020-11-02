a0 = 0.010
amax = 0.025
H = 15.0
h = 3.0
Vp = 1e-9
rho = 2.670

function boundary(x, y, t)
    return Vp/2.0 * t
end

function mu(x, y)
    return 32.038120320
end

function eta(x, y)
    return math.sqrt(mu(x, y) * rho) / 2.0
end

function Vinit(x, y)
    return 1.0e-9
end

function a(x, y)
    local z = -y
    if z < H then
        return a0
    elseif z < H + h then
        return a0 + (amax - a0) * (z - H) / h
    else
        return amax
    end
end

function asinh(x)
    return math.log(x + math.sqrt(x^2 + 1))
end

function tau_pre(x, y)
    local f0 = 0.6
    local b = 0.015
    local V0 = 1e-6
    local Vi = Vinit(x, y)
    local sn = 50.0
    local e = math.exp((f0 + b * math.log(V0 / Vi)) / amax);
    return sn * amax * asinh((Vi / (2.0 * V0)) * e) + eta(x, y) * Vi;
end
