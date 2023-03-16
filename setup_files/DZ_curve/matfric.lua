local BP1 = {}

-- Frictional parameters
BP1.a_b1 = 0.012
BP1.a_b2 = -0.004
BP1.a_b3 = 0.015
BP1.a_b4 = 0.024
BP1.b0 = 0.019

-- Shear stress [MPa]: negative for right-lateral
BP1.tau1 = -10
BP1.tau2 = -30
BP1.tau3 = -22.5

-- Normal stress [MPa]: positive for compression 
BP1.sig1 = 10
BP1.sig2 = 50

-- Depths where parameters vary [km]
BP1.Wf = 24
BP1.H = 12.0
BP1.h = 5.0
BP1.H2 = 2.0

-- DZ-related parameters (r = 1: DZ, 2: elsewhere)
BP1.fzd = 10.5          -- Damage zone depth [km]
BP1.mu_default = 32     -- Default shear modulus [GPa]
BP1.mu_damage = 10      -- DZ shear modulus [GPa]

-- Others
BP1.Vp = 1e-9           -- Plate rate [m/s]
BP1.rho0 = 2.670        -- Density []
BP1.V0 = 1.0e-6         -- Reference slip rate [m/s]
BP1.f0 = 0.6            -- Reference friction coefficient

function BP1:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function BP1:boundary(x, y, t, r)
    if x > 1.0 then
        return self.Vp/2.0 * t
    elseif x < -1.0 then
        return -self.Vp/2.0 * t
    else
        return self.Vp * t
    end
end

function BP1:mu(x, y, r)
	-- print("BP1:mu(x, y, r) -> ", r)
    if r == 1.0 then
        print("BP1: r, mu ->",r,self.mu_damage)
        return self.mu_damage
    else
        print("BP1: r, mu ->",r,self.mu_default)
        return self.mu_default
    end
end

function BP1:eta(x, y, r)
	-- print("BP1:eta(y, r) -> (", y, ",", r, ")")
    local z = -y
    if z < self.fzd then
        return math.sqrt(self.mu_damage * self.rho0) / 2.0
    else
        return math.sqrt(self.mu_default * self.rho0) / 2.0
    end
end

function BP1:L(x, y, r)
    return 0.008
end

function BP1:sn_pre(x, y, r)
    local z = -y
    local _sigma1 = self.sig2 + (self.sig2 - self.sig1) * (z - self.H2) / self.H2
    if z < self.H2 then
        return _sigma1
    else
        return self.sig2
    end
end

function BP1:Vinit(x, y, r)
    return 1.0e-9
end

function BP1:ab(x, y, r)
    local z = -y
    local _ab1 = self.a_b2 + (self.a_b2 - self.a_b1) * (z - self.H2) / self.H2
    local _ab2 = self.a_b2 + (self.a_b3 - self.a_b2) * (z - self.H) / self.h
    local _ab3 = self.a_b3 + (self.a_b4 - self.a_b3) * (z - self.h - self.H) / (self.Wf - self.h - self.H)

    if z < self.H2 then
        return _ab1
    elseif z < self.H then
        return self.a_b2
    elseif z < self.H + self.h then
        return _ab2
    elseif z < self.Wf then
        return _ab3
    else
        return self.a_b4
    end
end

function BP1:b(x, y, r)
    return self.b0
end

function BP1:a(x, y, r)
    local z = -y
    local _ab = self:ab(x, y, r)
    local _b = self:b(x, y, r)
    return _ab + _b
end

function BP1:tau_pre(x, y, r)
    local z = -y
    local _tau1 = self.tau2 + (self.tau2 - self.tau1) * (z - self.H2) / self.H2
    local _tau2 = self.tau2 + (self.tau3 - self.tau2) * (z - self.H) / self.h

    if z < self.H2 then
        return _tau1
    elseif z < self.H then
        return self.tau2
    elseif z < self.H + self.h then
        return _tau2
    else
        return self.tau3
    end
end

bp1 = BP1:new()

bp1_sym = BP1:new()
function bp1_sym:boundary(x, y, t, r)
    return self.Vp/2.0 * t
end

