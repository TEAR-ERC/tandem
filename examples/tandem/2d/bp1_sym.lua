a0 = 0.010
amax = 0.025
H = 15.0
h = 3.0
f0 = 0.6
b = 0.015
V0 = 1e-6
Vp = 1e-9
Vinit = 1e-9
eta = 4.62444e0
sn = 50.0

function boundary(x, y, t)
    V = -Vp/2.0
    if x > 1.0 then
        V = Vp/2.0
    end
    return V * t
end

function mu(x, y)
    return 32.038120320
end

function a(x, y)
    z = -y
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

function tau0(x, y)
    e = math.exp((f0 + b * math.log(V0 / Vinit)) / amax);
    return sn * amax * asinh((Vinit / (2.0 * V0)) * e) + eta * Vinit;
end
