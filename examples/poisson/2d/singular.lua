function solution(x, y)
    local r = math.sqrt(x^2 + y^2)
    local phi = math.atan2(y, x)
    if phi < 0 then
        phi = phi + 2 * math.pi
    end
    local delta = 0.5354409456;
    local a = {0.4472135955, -0.7453559925, -0.9441175905, -2.401702643}
    local b = {1.0, 2.333333333, 0.55555555555, -0.4814814814}
    local dNo = 1;
    if x < 0 and y > 0 then
        dNo = 2
    elseif x < 0 and y < 0 then
        dNo = 3
    elseif x > 0 and y < 0 then
        dNo = 4
    end
    return r^delta * (a[dNo] * math.sin(delta * phi) + b[dNo] * math.cos(delta * phi))
end

function coefficient(x, y)
    if x * y >= 0 then
        return 5.0
    else
        return 1.0
    end
end
