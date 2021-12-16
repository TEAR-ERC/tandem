local Embedded = {}

function Embedded:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function Embedded:solution(x, y)
    if x < 0.5 then
        return 3.0
    else
        return 2.0
    end
end

function Embedded:slip(x, y)
    return 1.0
end

function Embedded:boundary(x, y)
    return self:solution(x, y)
end

embedded = Embedded:new()
