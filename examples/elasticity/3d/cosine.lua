require "warp"

function mu(x, y, z)
    return 1.0
end

function lam(x, y, z)
    return 2.0
end

function force(x, y, z)
    return math.pi*(-3.0*math.pi*math.sin(math.pi*x)*math.sin(math.pi*z) + 5.0*math.pi*math.cos(math.pi*x)*math.cos(math.pi*y)), -3.0*math.pi^2*math.sin(math.pi*x)*math.sin(math.pi*y), 5.0*math.pi^2*math.cos(math.pi*x)*math.cos(math.pi*z)
end

function solution(x, y, z)
    return math.cos(math.pi * x) * math.cos(math.pi * y),
           0,
           math.cos(math.pi * x) * math.cos(math.pi * z)
end

function solution_jacobian(x, y, z)
    return -math.pi * math.sin(math.pi * x) * math.cos(math.pi * y),
           -math.pi * math.cos(math.pi * x) * math.sin(math.pi * y),
           0,
           0, 0, 0,
           -math.pi * math.sin(math.pi * x) * math.cos(math.pi * z),
           0,
           -math.pi * math.cos(math.pi * x) * math.sin(math.pi * z)
end
