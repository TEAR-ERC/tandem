-- Benchmark derived from https://pylith.readthedocs.io/en/latest/user/benchmarks/savage-prescott.html

local SavagePrescott = {}

SavagePrescott.H_locked = 20
SavagePrescott.H_creep = 40.0
SavagePrescott.Vp = 2 * 1e-2 / (365.25 * 24 * 3600) -- 2 cm/year in m/s
SavagePrescott.V0 = 0

SavagePrescott.rho0 = 2.700
SavagePrescott.cs = 3.333333
SavagePrescott.b = 0.0003
SavagePrescott.T = 200 * 365.25 * 24 * 3600 -- 200 years in seconds
SavagePrescott.theta = 0.2
SavagePrescott.f0 = 0.1

function SavagePrescott:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function SavagePrescott:lam(x, y, z)
    -- Lame parameter for Poisson solid (nu = 0.25)
    local mu = 30.0
    local nu = 0.25
    return 30.0
end

function SavagePrescott:boundary(x, y, z, t)
    local slip = 0.0
    local cycle_time = t % self.T -- Time within current cycle (0 to T)
    local cycle_number = math.floor(t / self.T) -- Which cycle we're in
    local accumulated_slip = cycle_number * self.Vp * self.T -- Total slip from previous earthquakes
    -- Dirichlet Surfaces
    if x > 1.0 then
        slip = -self.Vp / 2.0 * t -- +1 cm/yr in y-direction
    elseif x < -1.0 then
        slip = self.Vp / 2.0 * t -- -1 cm/yr in y-direction
        -- Fault surfaces
    elseif x > -1.0 and x < 1.0 then -- Near the fault (x ≈ 0)
        if math.abs(z + 20) < 0.3 then
            -- This is essentially the average of coseismic slip and creep but 
            -- This is not always applied well because the geometry does not always have a node at z = -20 km, so it is applied to the node closest to z = -20 km. Removing this also works but trying to keep it consistent with the Pylith benchmark.
            slip = accumulated_slip / 2.0 + (self.Vp / 2.0) * t
        elseif z > -self.H_locked and z <= 0 then
            slip = accumulated_slip
            -- Creeping section: -40 km < z < -20 km (steady creep)
        elseif z < -self.H_locked and z >= -self.H_creep then
            -- Continuous creep at full plate rate (2 cm/yr)
            slip = self.Vp * t

        else
            -- Never called - should not be called
            slip = 0.0
        end
    end

    return 0.0, slip, 0.0
end

function SavagePrescott:mu0(x, y, z)
    if z >= -self.H_creep then
        return 30.0 -- Purely elastic part
    end
    return 0.0

end

-- stiffness of viscoelastic branch of the maxwell element
function SavagePrescott:mu1(x, y, z)
    if z >= -self.H_creep then
        return 0.0 -- We need to make sure this is numercally stable
    end
    return 30.0

end

function SavagePrescott:free_slip_boundary(x, y, z, t)
    return 0.0
end

function SavagePrescott:viscosity(x, y, z)
    return 2.36682e+10

end

-- eta/mu1
function SavagePrescott:relaxation_time(x, y, z)
    return SavagePrescott:viscosity(x, y, z) / 30.0
end

savage_prescott = SavagePrescott:new()

