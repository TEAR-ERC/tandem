local PW = {}
PW.__index = PW

PW.b = 0.012
PW.V0 = 1.0e-6
PW.f0 = 0.6
PW.Vini = 1e-12

PW.rho0 = 2.670
PW.cs = 3.464
PW.nu = 0.25

PW.k_i = 5.0

function PW.new()
    local self = setmetatable({}, PW)
    return self
end

function PW:rho(x, y, z)
    return self.rho0
end

function PW:mu(x, y, z)
    return self.cs^2 * self.rho0
end

function PW:lam(x, y, z)
    return 2 * self.nu * self:mu(x,y) / (1 - 2 * self.nu)
end

function PW:c_p(x, y, z)
    return math.sqrt((self:lam(x, y, z) + 2.0 * self:mu(x, y, z))/self.rho0)
end

function PW:solution(x, y, z, t)
    local w = self:c_p(x, y, z) * self.k_i * math.sqrt(3)
    local u_i = -self.k_i * math.sin(self.k_i * (x + y + z) - w * t)
    return u_i, u_i, u_i
end

function PW:initial_displacement(x, y, z)
    return self:solution(x, y, z, 0.0)
end

function PW:initial_velocity(x, y, z)
    local w = self:c_p(x, y, z) * self.k_i * math.sqrt(3)
    local v_i = w * self.k_i * math.cos(self.k_i * (x + y + z))
    return v_i, v_i, v_i
end

function PW:boundary(x, y, z, t)
    return self:solution(x, y, z, t)
end


plane_wave = PW.new()
