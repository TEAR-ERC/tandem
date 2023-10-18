require "cosine"

local CosineVariable = Cosine:new()

function CosineVariable:mu(x, y)
    return (x+y)^3
end

function CosineVariable:lam(x, y)
    return (x+y)^4
end

function CosineVariable:force(x, y)
    return math.pi*(math.pi*(x + y)^3*(1.0*x + 1.0*y + 2.0)*math.cos(math.pi*x)*math.cos(math.pi*y) + 1.0*math.pi*(x + y)^3*math.cos(math.pi*x)*math.cos(math.pi*y) + 3.0*(x + y)^2*math.sin(math.pi*y)*math.cos(math.pi*x) + (4.0*(x + y)^3 + 6.0*(x + y)^2)*math.sin(math.pi*x)*math.cos(math.pi*y)), -1.0*math.pi*(math.pi*(x + y)^4*math.sin(math.pi*x)*math.sin(math.pi*y) + math.pi*(x + y)^3*math.sin(math.pi*x)*math.sin(math.pi*y) - 4*(x + y)^3*math.sin(math.pi*x)*math.cos(math.pi*y) - 3*(x + y)^2*math.sin(math.pi*y)*math.cos(math.pi*x))
end

cosine_variable = CosineVariable:new()
