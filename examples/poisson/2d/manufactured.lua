require "warp"

function warp(x, y)
    xp, yp = partialAnnulus(x, y)
    return math.exp(xp) * math.sin(xp) * xp, math.exp(yp) * yp
end

function solution(x, y)
    return math.exp(-x - y^2)
end

function force(x, y)
    return (1.0 - 4.0 * y^2) * math.exp(-x - y^2)
end
