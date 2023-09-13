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
BP1.fzw = 0.5           -- Damage zone width [km]
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

function BP1:mu(x, y, r)
    local z = -y
    local region = 2.
    
    if x <= self.fzw then
        if z <= self.fzd-self.fzw then
            -- return self.mu_damage
            region = 1.
        elseif z <= self.fzd then
            if x <= math.sqrt(self.fzw^2 - (z-(self.fzd-self.fzw))^2) then
                -- return self.mu_damage
                region = 1.
            end
        end
    end
    print('region =',region)

    if region == 1. then
        return self.mu_damage
    elseif region == 2. then
        return self.mu_default
    end
end

bp1 = BP1:new()

bp1_sym = BP1:new()
function bp1_sym:boundary(x, y, t, r)
    return self.Vp/2.0 * t
end

local _x = 0
local _y = -7 
print('x, y ->',_x,_y)
num = BP1:mu(_x,_y,0)
print(num)