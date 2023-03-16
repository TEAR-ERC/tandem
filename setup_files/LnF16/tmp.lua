local BP1 = {}

BP1.a0 = 0.02
BP1.amax = 0.023
BP1.a_b0 = 0.007
BP1.a_bmin = -0.003
BP1.a_bmax = 0.023
BP1.a_b1 = 0.015
BP1.H = 9.0
BP1.h = 6.0
BP1.H2 = 2.0
BP1.Wf = 24
BP1.Vp = 1e-9
BP1.rho0 = 2.670
BP1.V0 = 1.0e-6
BP1.f0 = 0.6

function BP1:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function BP1:boundary(x, y, t)
    if x > 1.0 then
        return self.Vp/2.0 * t
    elseif x < -1.0 then
        return -self.Vp/2.0 * t
    else
        return self.Vp * t
    end
end

function BP1:mu(x, y)
    return 32.038120320
end

function BP1:eta(x, y)
    return math.sqrt(self:mu(x, y) * self.rho0) / 2.0
end

function BP1:L(x, y)
    -- Or Dc in other articles
    return 0.004
end

function BP1:sn_pre(x, y)
    return 50.0
end

function BP1:Vinit(x, y)
    return 1.0e-9
end

function BP1:a(x, y)
    local z = -y
    local _a = self.a0 + (self.amax - self.a0) * (z - self.H - self.h) / (self.Wf - self.h - self.H)
    if z < self.H + self.h then
        -- print('z',z,'/ upper', self.a0 )
        return self.a0
    elseif z < self.Wf then
        -- print('z',z,'/ middle', _a)
        return _a
    else
        -- print('z',z,'/ lower',self.amax)
        return self.amax
    end
end

function BP1:b(x, y)
    return 0.015
end

-- function BP1:b(x, y)
--     local z = -y
--     local _a = self:a(x,y)
--     print(_a)
--     return 0.014
--     -- if z < self.H2 then
--     --     print('z',z,'uppermost', self.a_b0)
--     --     return _a - self.a_b0
--     -- elseif z < self.H then
--     --     print('z',z,'middle', self.a_bmin)
--     --     return _a - self.a_bmin
--     -- elseif z < self.H + self.h then
--     --     local _b = _a - self.a_bmin + (self.a_b1 - self.a_bmin) * (z - self.H) / self.h
--     --     print('z',z,'slant1', _ab)
--     -- elseif z < self.Wf then
--     --     local _b = _a - self.a_b1 + (self.a_bmax - self.a_b1) * (z - self.H - self.h) / (self.Wf - self.h - self.H)
--     --     print('z',z,'slant2', _ab)
--     -- else
--     --     print('z',z,'lower', self.a_bmax)
--     --     return _a - self.a_bmax
--     -- end
--     -- return _b
-- end

-- function BP1:ab(x, y)
--     local z = -y
--     local _ab1 = self.a_bmin + (self.a_b1 - self.a_bmin) * (z - self.H) / self.h
--     print(_ab1)
--     local _ab2 = self.a_b1 + (self.a_bmax - self.a_b1) * (z - self.H - self.h) / (self.Wf - self.h - self.H)
--     print(_ab2)
--     if z < self.H2 then
--         print('z',z,'uppermost', self.a_b0)
--         return self.a_b0
--     elseif z < self.H then
--         print('z',z,'middle', self.a_bmin)
--         return self.a_bmin
--     elseif z < self.H + self.h then
--         print('z',z,'slant1', _ab1)
--         return _ab1
--     elseif z < self.Wf then
--         print('z',z,'slant2', _ab2)
--         return _ab2
--     else
--         print('z',z,'lower', self.a_bmax)
--         return self.a_bmax
--     end
-- end

-- function BP1:b(x, y)
-- --     local _a = self:a(x,y)
-- --     print('a:',_a)
-- --     local _ab = self:ab(x,y)
-- --     print('a-b:',_ab)
-- --     print("z, a, a-b: ",z,_a, _ab)
--     -- return _a - _ab
--     return 0.015
-- end

-- function BP1:tau_pre(x, y)
--     local Vi = self:Vinit(x, y)
--     local sn = self:sn_pre(x, y)
--     local _b = self:b(x, y)
--     print('x,y,b:',x,y,_b)
--     local e = math.exp((self.f0 + _b * math.log(self.V0 / Vi)) / self.amax)
--     return -(sn * self.amax * math.asinh((Vi / (2.0 * self.V0)) * e) + self:eta(x, y) * Vi)
-- end


function BP1:tau_pre(x, y)
    local Vi = self:Vinit(x, y)
    local sn = self:sn_pre(x, y)
    local e = math.exp((self.f0 + 0.015 * math.log(self.V0 / Vi)) / self.amax)
    return -(sn * self.amax * math.asinh((Vi / (2.0 * self.V0)) * e) + self:eta(x, y) * Vi)
end

-- BP1.b = 0.015
-- function BP1:tau_pre(x, y)
--     local Vi = self:Vinit(x, y)
--     local sn = self:sn_pre(x, y)
--     local e = math.exp((self.f0 + self.b * math.log(self.V0 / Vi)) / self.amax)
--     return -(sn * self.amax * math.asinh((Vi / (2.0 * self.V0)) * e) + self:eta(x, y) * Vi)
-- end

bp1 = BP1:new()

bp1_sym = BP1:new()
function bp1_sym:boundary(x, y, t)
    return self.Vp/2.0 * t
end

