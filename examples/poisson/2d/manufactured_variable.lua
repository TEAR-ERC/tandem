require "manufactured"

function force(x, y)
    return (1.0 + 3.0 * y - 4.0 * y^3 + x - 4.0 * x * y^2) * math.exp(-x - y^2)
end

function coefficient(x, y)
    return x + y
end
