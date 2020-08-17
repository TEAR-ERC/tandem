require "warp"

function warp(x, y)
    return partialAnnulus(x, y)
end

function solution(x, y)
    return math.exp(-x - y^2)
end

function force(x, y)
    return (1.0 - 4.0 * y^2) * math.exp(-x - y^2)
end
