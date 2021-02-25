require "cosine"

function warp(x, y)
    xp, yp = partialAnnulus(x, y)
    return math.exp(xp) * math.sin(xp) * xp, math.exp(yp) * yp
end

local freq = 10.0

function mu(x, y)
    return math.sin(freq*math.pi*x)
end

function lam(x, y)
    return math.sin(freq*math.pi*y)
end

function force(x, y)
    return math.pi*(2.0*math.pi*freq*math.sin(math.pi*x)*math.cos(math.pi*y)*math.cos(math.pi*freq*x) + math.pi*(2.0*math.sin(math.pi*freq*x) + 1.0*math.sin(math.pi*freq*y))*math.cos(math.pi*x)*math.cos(math.pi*y) + 1.0*math.pi*math.sin(math.pi*freq*x)*math.cos(math.pi*x)*math.cos(math.pi*y)), 1.0*math.pi*(math.pi*freq*math.sin(math.pi*x)*math.cos(math.pi*y)*math.cos(math.pi*freq*y) + math.pi*freq*math.sin(math.pi*y)*math.cos(math.pi*x)*math.cos(math.pi*freq*x) - math.pi*math.sin(math.pi*x)*math.sin(math.pi*y)*math.sin(math.pi*freq*x) - math.pi*math.sin(math.pi*x)*math.sin(math.pi*y)*math.sin(math.pi*freq*y))
end
