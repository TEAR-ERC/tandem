EmbeddedHalf = {}

function EmbeddedHalf:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

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

function EmbeddedHalf:force(x, y)
    if y > 0.0 then
        return 0, -2 * sign(x) * dsmoothstep_dx2(y)
    else
        return 0, 0
    end
end

function EmbeddedHalf:slip(x, y)
    if y > 0.0 then
        return 0, 2 * smoothstep(y)
    else
        return 0, 0
    end
end

function EmbeddedHalf:solution(x, y)
    if y > 0.0 then
        return 0, sign(x) * smoothstep(y)
    else
        return 0, 0
    end
end

function EmbeddedHalf:solution_jacobian(x, y)
    if y > 0.0 then
        return 0, 0,
               0, sign(x) * dsmoothstep_dx(y)
    else
        return 0, 0,
               0, 0
    end
end

function EmbeddedHalf:boundary(x, y)
    return self:solution(x, y)
end

function EmbeddedHalf:lam(x, y)
    return 0
end

embedded_half = EmbeddedHalf:new()
