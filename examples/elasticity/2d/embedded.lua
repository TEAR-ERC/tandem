function smoothstep(x)
    return -20 * x^7 + 70 * x^6 - 84 * x^5 + 35 * x^4
end

function dsmoothstep_dx(x)
    return -140 * x^6 + 420 * x^5 - 420 * x^4 + 140 * x^3
end

function dsmoothstep_dx2(x)
    return -840 * x^5 + 2100 * x^4 - 1680 * x^3 + 420 * x^2
end

function sign(x)
    if x > 0.5 then
        return -1.0
    else
        return 1.0
    end
end

function force(x, y)
    if -1 < y and y < 1 then
        return 0, 2 * sign(x) * dsmoothstep_dx2(math.abs(y))
    else
        return 0, 0
    end
end

function slip(x, y)
    return 0.0, 2 * (1-smoothstep(math.abs(y)))
end

function solution(x, y)
    if -1 < y and y < 1 then
        return 0, sign(x) * (1-smoothstep(math.abs(y)))
    else
        return 0, 0
    end
end

function solution_jacobian(x, y)
    if -1 < y and y < 1 then
        return 0, 0,
               0, -sign(x) * dsmoothstep_dx(math.abs(y)) * (y / math.abs(y))
    else
        return 0, 0,
               0, 0
    end
end

function lam(x, y)
    return 0
end
