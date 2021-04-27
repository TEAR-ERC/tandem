require "manufactured"

function force(x, y)
    return (-4*y^2*(x + y)^5 + 10*y*(x + y)^4 + (x + y)^5 + 5*(x + y)^4)*math.exp(-x - y^2)
end

function coefficient(x, y)
    return (x + y)^5
end
