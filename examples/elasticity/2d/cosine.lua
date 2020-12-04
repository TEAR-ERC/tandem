function mu(x, y)
    return 1.0
end

function lam(x, y)
    return 1.0
end

function force(x, y)
    return (lam(x, y) + 3.0*mu(x, y)) * math.pi^2 * math.cos(math.pi * x) * math.cos(math.pi * y),
           -(lam(x, y) + mu(x, y)) * math.pi^2 * math.sin(math.pi * x) * math.sin(math.pi * y)
end

function solution(x, y)
    return math.cos(math.pi * x) * math.cos(math.pi * y), 0
end

