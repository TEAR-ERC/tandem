local delta = 0.5354409456;
local a = {0.4472135955, -0.7453559925, -0.9441175905, -2.401702643}
local b = {1.0, 2.333333333, 0.55555555555, -0.4814814814}

function region(x, y)
    local dNo = 1
    if x < 0 and y > 0 then
        dNo = 2
    elseif x < 0 and y < 0 then
        dNo = 3
    elseif x > 0 and y < 0 then
        dNo = 4
    end
    return dNo
end

function polar(x, y)
    local r = math.sqrt(x^2 + y^2)
    local phi = math.atan2(y, x)
    if phi < 0 then
        phi = phi + 2 * math.pi
    end
    return r, phi
end

function solution(x, y)
    local r, phi = polar(x, y)
    local dNo = region(x, y)
    return r^delta * (a[dNo] * math.sin(delta * phi) + b[dNo] * math.cos(delta * phi))
end

function solution_jacobian(x, y)
    local r, phi = polar(x, y)
    local dNo = region(x, y)
    drdx = x / r
    drdy = y / r
    dphidx = -y / r^2
    dphidy =  x / r^2
    dsdr = delta * r^(delta-1) *
            (a[dNo] * math.sin(delta * phi) + b[dNo] * math.cos(delta * phi))
    dsdphi = delta * r^delta *
            (a[dNo] * math.cos(delta * phi) - b[dNo] * math.sin(delta * phi))
    return dsdr * drdx + dsdphi * dphidx,
           dsdr * drdy + dsdphi * dphidy
end

function coefficient(x, y)
    if x * y >= 0 then
        return 5.0
    else
        return 1.0
    end
end
