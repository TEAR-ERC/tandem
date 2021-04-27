function smoothstep(x)
    return 6 * x^5 - 15 * x^4 + 10 * x^3
end

function dsmoothstep_dx2(x)
    return 120 * x^3 - 180 * x^2 + 60 * x
end

function sign(x)
    if x > 0.5 then
        return -1.0
    else
        return 1.0
    end
end

function force(x, y)
    if y > 0.0 then
        return -sign(x) * dsmoothstep_dx2(y)
    else
        return 0
    end
end

function solution(x, y)
    if y > 0.0 then
        return sign(x) * smoothstep(y)
    else
        return 0.0
    end
end

function slip(x, y)
    if y > 0.0 then
        return 2.0 * smoothstep(y)
    else
        return 0.0
    end
end
