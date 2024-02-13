local BP6 = {}

-- Fluid injection related parameters
BP6.alpha = 0.1             -- m2/s
BP6.beta = 1e-8             -- Pa-1
BP6.phi = 0.1
BP6.q0 = 1.25e-6            -- m/s
BP6.toff = 100*60*60*24     -- s

-- Rate-and-state related parameters
BP6.b = 0.005       -- Evolution effect parameter
BP6.V0 = 1.0e-6     -- Reference slip rate
BP6.f0 = 0.6        -- Reference F.C. 

-- Rate-and-state related parameters
BP6.lf = 20         -- Fault half length [km]
BP6.rho0 = 2.670    -- Density
BP6.cs = 3.464      -- Shear wave speed
BP6.nu = 0.25       -- Poisson ratio

-- Define numerical approximation of the error function (A&S formula 7.1.26)
function erf(x)
    -- constants
    local a1 =  0.254829592
    local a2 = -0.284496736
    local a3 =  1.421413741
    local a4 = -1.453152027
    local a5 =  1.061405429
    local p  =  0.3275911

    local sign = 1
    if x < 0 then
        sign = -1
    end
    local x = math.abs(x)

    local c = 1.0/(1.0 + p*x)
    local val = 1.0 - (((((a5*c + a4)*c) + a3)*c + a2)*c + a1)*c*math.exp(-x*x)

    return sign*val
end

-- Define G function from BP6 description, eq. (27)
function getG(y, t, alpha)
    local z = y*1e3
    local Gterm1 = 0.0
    local Gterm2 = 0.0
    local G = 0.0
    if t > 0 then
        Gterm1 = math.exp(-(z*z)/(4*alpha*t)) / math.sqrt(math.pi)
        Gterm2 = (math.abs(z)/math.sqrt(4*alpha*t))*(1-erf(math.abs(z)/math.sqrt(4*alpha*t)))
        G = math.sqrt(t)*(Gterm1-Gterm2)
    end
    return G
end

function BP6:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function BP6:boundary(x, y, t)
    if math.abs(y) > self.lf then
        return 0
    end
end

function BP6:mu(x, y)
    return 32.04
end

function BP6:eta(x, y)
    return self:mu(x, y) / ( 2 * self.cs )
end

function BP6:L(x, y)
    return 0.004
end

function BP6:sn_pre(x, y)
    return 50.0
end

function BP6:delta_sn(x, y, t)
    local q = 0.0
    local mult = self.q0/(self.beta * self.phi * math.sqrt(self.alpha))*1e-6 -- turn to MPa unit
    if t >= 0 then
        Gt = getG(y, t, self.alpha)
        if t <= self.toff then
            q = mult*Gt
        else
            Gttoff = getG(y, t - self.toff, self.alpha)
            q = mult*(Gt - Gttoff)
        end
    end
    return q
end

function BP6:Vinit(x, y)
    return 1.0e-12
end

function BP6:a(x, y)
    return 0.007
end

function BP6:tau_pre(x, y)
    return -29.2
end

function BP6:lam(x, y)
    return 2 * self.nu * self:mu(x,y) / (1 - 2 * self.nu)
end

bp6 = BP6:new()