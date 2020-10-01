function force(x, y)
    return 4 * math.pi^2 * math.cos(math.pi * x) * math.cos(math.pi * y),
           -2 * math.pi^2 * math.sin(math.pi * x) * math.sin(math.pi * y)
end

function solution(x, y)
    return math.cos(math.pi * x) * math.cos(math.pi * y), 0
end

