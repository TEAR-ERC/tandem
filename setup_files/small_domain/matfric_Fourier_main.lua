local BP1 = {}

-- Frictional parameters
BP1.a_b1 = 0.015
BP1.a_b2 = -0.004
BP1.b = 0.019

-- Shear stress [MPa]: negative for right-lateral
BP1.tau1 = -10
BP1.tau2 = -30

-- Normal stress [MPa]: positive for compression
BP1.sigma1 = 10
BP1.sigma2 = 50

-- Depths where parameters vary [km]
BP1.Wf = 24
BP1.H = 12.0
BP1.H2 = 2.0

-- Others
BP1.Vp = 1e-9           -- Plate rate [m/s]
BP1.rho0 = 2.670        -- Density []
BP1.V0 = 1.0e-6         -- Reference slip rate [m/s]
BP1.f0 = 0.6            -- Reference friction coefficient

-------------------- Define your domain: below is the base values for all the parameters
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
    return 10.0
end

function BP1:eta(x, y)
    return math.sqrt(self:mu(x, y) * self.rho0) / 2.0
end

function BP1:L(x, y)
    return 0.004
end

function BP1:sn_pre(x, y)
    local z = -y
    local _sigma1 = self.sigma2 + (self.sigma2 - self.sigma1) * (z - self.H2) / self.H2
    if z < self.H2 then
        return _sigma1
    else
        return self.sigma2
    end
end

function BP1:Vinit(x, y)
    return 1.0e-9
end

function BP1:ab(x, y)
    local z = -y
    local _ab1 = self.a_b2 + (self.a_b2 - self.a_b1) * (z - self.H2) / self.H2
    local _ab3 = self.a_b2 + (self.a_b1 - self.a_b2) * (z - self.H) / (self.Wf - self.H)

    if z < self.H2 then
        return _ab1
    elseif z < self.H then
        return self.a_b2
    elseif z < self.Wf then
        return _ab3
    else
        return self.a_b1
    end
end

function BP1:a(x, y)
    local z = -y
    local _ab = self:ab(x,y)
    return _ab + self.b
end

function BP1:tau_pre(x, y)
    local z = -y
    local _tau1 = self.tau2 + (self.tau2 - self.tau1) * (z - self.H2) / self.H2
    if z < self.H2 then
        return _tau1
    else
        return self.tau2
    end
end

bp1 = BP1:new()

bp1_sym = BP1:new()
function bp1_sym:boundary(x, y, t)
    return self.Vp/2.0 * t
end

