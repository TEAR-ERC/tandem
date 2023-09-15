require "cosine"

local CosineOscillatory = Cosine:new()

local freq = 10.0

function CosineOscillatory:warp(x, y)
    return x, y
end

function CosineOscillatory:mu(x, y)
    return 1 + 0.25 * math.sin(freq*math.pi*x)
end

function CosineOscillatory:lam(x, y)
    return 1 + 0.25 * math.sin(freq*math.pi*y)
end

function CosineOscillatory:force(x, y)
    return math.pi*(0.5*math.pi*freq*math.sin(math.pi*x)*math.cos(math.pi*y)*math.cos(math.pi*freq*x) + 1.0*math.pi*(0.25*math.sin(math.pi*freq*x) + 1)*math.cos(math.pi*x)*math.cos(math.pi*y) + math.pi*(0.5*math.sin(math.pi*freq*x) + 0.25*math.sin(math.pi*freq*y) + 3.0)*math.cos(math.pi*x)*math.cos(math.pi*y)), -1.0*math.pi*(-0.25*math.pi*freq*math.sin(math.pi*x)*math.cos(math.pi*y)*math.cos(math.pi*freq*y) - 0.25*math.pi*freq*math.sin(math.pi*y)*math.cos(math.pi*x)*math.cos(math.pi*freq*x) + math.pi*(0.25*math.sin(math.pi*freq*x) + 1)*math.sin(math.pi*x)*math.sin(math.pi*y) + math.pi*(0.25*math.sin(math.pi*freq*y) + 1)*math.sin(math.pi*x)*math.sin(math.pi*y))
end

cosine_oscillatory = CosineOscillatory:new()
