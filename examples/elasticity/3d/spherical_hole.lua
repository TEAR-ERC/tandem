local S = 1.0
local a = 0.5
local mu0 = 1.0
local nu = 0.25

local SphericalHole = {}

function SphericalHole:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function SphericalHole:mu(x, y, z)
    return mu0
end

function SphericalHole:lam(x, y, z)
    return 2.0 * mu0 * nu / (1-2*nu)
end

function SphericalHole:force(x, y, z)
    return 0, 0, 0
end

function spherical(x, y, z)
    local R = math.sqrt(x^2 + y^2 + z^2)
    local theta = math.atan2(y, x)
    local beta = math.acos(z / R)
    return R, theta, beta
end

function u_spherical(R, theta, beta)
    local x0 = nu + 1
    local x1 = a^5
    local x2 = R^5
    local x3 = 5*nu - 7
    local x4 = x2*x3
    local x5 = 10*nu
    local x6 = R^2*a^3
    local x7 = 2*beta
    local x8 = S/(R^4*mu0*x3)
    local uR = (1.0/8.0)*x8*(x0*x6*(x5 - 13) + x0*(9*x1 + x2*(x5 - 14) + x6*(20*nu - 25))*math.cos(x7) + x1*(3*nu + 3) + 2*x4*(1 - nu))/x0
    local ub = -1.0/4.0*x8*(-3*x1 + x4 + x6*(x5 - 5))*math.sin(x7)
    return uR, ub
end

function SphericalHole:solution(x, y, z)
    R, theta, beta = spherical(x, y, z)
    local uR, ub = u_spherical(R, theta, beta)

    local ct = math.cos(theta)
    local st = math.sin(theta)
    local cb = math.cos(beta)
    local sb = math.sin(beta)
    return uR * ct * sb + ub * ct * cb,
           uR * st * sb + ub * st * cb,
           uR * cb      - ub * sb
end

function SphericalHole:solution_jacobian(x, y, z)
    R, theta, beta = spherical(x, y, z)
    local uR, ub = u_spherical(R, theta, beta)

    local x0 = nu + 1
    local x1 = a^5
    local x2 = R^5
    local x3 = 5*nu
    local x4 = x3 - 7
    local x5 = x2*x4
    local x6 = 10*nu
    local x7 = R^2*a^3
    local x8 = 2*beta
    local x9 = math.cos(x8)
    local x10 = 20*nu
    local x11 = x7*(x10 - 25)
    local x12 = S/(mu0*x4)
    local x13 = (1.0/4.0)*x12
    local x14 = x13/x2
    local x15 = R^(-4)
    local x16 = math.sin(x8)
    local uR_R = -x14*(x0*x7*(x6 - 13) + x0*x9*(18*x1 + x11 - x5) + x1*(6*nu + 6) + x5*(nu - 1))/x0
    local uR_b = -x13*x15*x16*(9*x1 + x11 + x2*(x6 - 14))
    local ub_R = x14*x16*(-12*x1 + x10*x7 - x2*x3 + 7*x2 - 10*x7)
    local ub_b = -1.0/2.0*x12*x15*x9*(-3*x1 + x5 + x7*(x6 - 5))

    local ct = math.cos(theta)
    local st = math.sin(theta)
    local cb = math.cos(beta)
    local sb = math.sin(beta)
    local R_x = ct*sb
    local R_y = st*sb
    local R_z = cb
    local t_x = -st / (R * sb)
    local t_y =  ct / (R * sb)
    local t_z = 0
    local b_x = ct * cb / R
    local b_y = st * cb / R
    local b_z = -sb / R
    local uR_x = uR_R * R_x + uR_b * b_x
    local uR_y = uR_R * R_y + uR_b * b_y
    local uR_z = uR_R * R_z + uR_b * b_z
    local ub_x = ub_R * R_x + ub_b * b_x
    local ub_y = ub_R * R_y + ub_b * b_y
    local ub_z = ub_R * R_z + ub_b * b_z

    local ux_x = uR_x*ct*sb + ub_x*ct*cb + uR*(-st*sb*t_x+ct*cb*b_x) - ub*(st*cb*t_x+ct*sb*b_x)
    local ux_y = uR_y*ct*sb + ub_y*ct*cb + uR*(-st*sb*t_y+ct*cb*b_y) - ub*(st*cb*t_y+ct*sb*b_y)
    local ux_z = uR_z*ct*sb + ub_z*ct*cb + uR*(-st*sb*t_z+ct*cb*b_z) - ub*(st*cb*t_z+ct*sb*b_z)
    local uy_x = uR_x*st*sb + ub_x*st*cb + uR*( ct*sb*t_x+st*cb*b_x) + ub*(ct*cb*t_x-st*sb*b_x)
    local uy_y = uR_y*st*sb + ub_y*st*cb + uR*( ct*sb*t_y+st*cb*b_y) + ub*(ct*cb*t_y-st*sb*b_y)
    local uy_z = uR_z*st*sb + ub_z*st*cb + uR*( ct*sb*t_z+st*cb*b_z) + ub*(ct*cb*t_z-st*sb*b_z)
    local uz_x = uR_x*cb - ub_x*sb - uR*sb*b_x - ub*cb*b_x
    local uz_y = uR_y*cb - ub_y*sb - uR*sb*b_y - ub*cb*b_y
    local uz_z = uR_z*cb - ub_z*sb - uR*sb*b_z - ub*cb*b_z

    return ux_x, ux_y, ux_z, uy_x, uy_y, uy_z, uz_x, uz_y, uz_z
end

function SphericalHole:boundary(x, y, z)
    return self:solution(x, y, z)
end

spherical_hole = SphericalHole:new()
