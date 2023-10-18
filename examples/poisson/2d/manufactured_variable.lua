require "manufactured"

local ManufacturedVariable = Manufactured:new()

function ManufacturedVariable:force(x, y)
    return (-4*y^2*(x + y)^5 + 10*y*(x + y)^4 + (x + y)^5 + 5*(x + y)^4)*math.exp(-x - y^2)
end

function ManufacturedVariable:mu(x, y)
    return (x + y)^5
end

manufactured_variable = ManufacturedVariable:new()
