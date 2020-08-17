require "warp"

function warp(x, y)
    return partialAnnulus(x, y)
end

local f = 10.0

function cos1D(x)
    return math.cos(f * math.pi * x)
end

function force(x, y)
    return 2 * f^2 * math.pi^2 * cos1D(x) * cos1D(y)
end

function solution(x, y)
    return cos1D(x) * cos1D(y)
end

