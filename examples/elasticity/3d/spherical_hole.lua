local S = 1.0
local a = 0.5
local mu0 = 1.0
local nu = 0.25

function mu(x, y, z)
    return mu0
end

function lam(x, y, z)
    return 2.0 * mu0 * nu / (1-2*nu)
end

function force(x, y, z)
    return 0, 0, 0
end

function spherical(x, y, z)
    local R = math.sqrt(x^2 + y^2 + z^2)
    local theta = math.atan2(y, x)
    local beta = math.acos(z / R)
    return R, theta, beta
end

function u_spherical(R, theta, beta)
    local x0 = 5*nu
    local x1 = x0 - 7
    local x2 = x1^3
    local x3 = 2*beta
    local x4 = 3*math.cos(x3) + 1
    local x5 = R^5
    local x6 = 2*nu*x5
    local x7 = nu + 1
    local x8 = x1^2
    local x9 = 3*a^5
    local x10 = a^3
    local x11 = x10*x7
    local x12 = R^2
    local x13 = 2*x12*x8
    local x14 = R^3*x1
    local x15 = S/(R^4*mu0*x7)
    local x16 = x10*x12
    local x17 = nu^2
    local uR = (1.0/8.0)*x15*(-x11*x13*(x0 - 6) - x13*(10*x11 + x14 - (4*nu - 3)*(5*x11 - x14))*math.cos(beta)^2 + x2*x4*x6 + x4*x7*x8*x9)/x2
    local ub = (1.0/4.0)*x15*(nu*x9 - x0*x16 - 10*x16*x17 + 5*x16 - 5*x17*x5 + 7*x5 + x6 + x9)*math.sin(x3)/x1
    return uR, ub
end

function solution(x, y, z)
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

function solution_jacobian(x, y, z)
    R, theta, beta = spherical(x, y, z)
    local uR, ub = u_spherical(R, theta, beta)

    local x0 = a^5
    local x1 = nu + 1
    local x2 = 2*beta
    local x3 = math.cos(x2)
    local x4 = 3*x3 + 1
    local x5 = 5*nu
    local x6 = x5 - 7
    local x7 = R^5
    local x8 = nu*x7
    local x9 = math.cos(beta)^2
    local x10 = a^3
    local x11 = R^2
    local x12 = x10*x11
    local x13 = 2*nu
    local x14 = 10*x10
    local x15 = S/(mu0*x6)
    local x16 = (1.0/4.0)*x15
    local x17 = x16/(x1*x7)
    local x18 = R^(-4)
    local x19 = 20*x12
    local x20 = math.sin(x2)
    local x21 = 7*x7
    local x22 = 12*x0
    local x23 = x11*x14
    local x24 = nu^2
    local x25 = nu*x23
    local uR_R = x17*(-6*x0*x1*x4 + 2*x1*x12*(x5 + 15*x9 - 6) + 2*x11*x9*(1 - x13)*(R^3*x6 + x1*x14) + x4*x6*x8)
    local uR_b = x16*x18*x20*(-nu*x19 - 9*x0 + 25*x12 + 14*x7 - 10*x8)
    local ub_R = x17*x20*(-nu*x22 + x13*x7 + x19*x24 + x21 - x22 - x23 - 5*x24*x7 + x25)
    local ub_b = (1.0/2.0)*x15*x18*x3*(3*x0 + 5*x12 + x21 - x25 - x5*x7)

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
    local uy_x = uR_x*st*sb + ub_x*st*cb + uR*( ct*sb*t_x+ct*cb*b_x) + ub*(ct*cb*t_x-ct*sb*b_x)
    local uy_y = uR_y*st*sb + ub_y*st*cb + uR*( ct*sb*t_y+ct*cb*b_y) + ub*(ct*cb*t_y-ct*sb*b_y)
    local uy_z = uR_z*st*sb + ub_z*st*cb + uR*( ct*sb*t_z+ct*cb*b_z) + ub*(ct*cb*t_z-ct*sb*b_z)
    local uz_x = uR_x*cb - ub_x*sb - uR*sb*b_x - ub*cb*b_x
    local uz_y = uR_y*cb - ub_y*sb - uR*sb*b_y - ub*cb*b_y
    local uz_z = uR_z*cb - ub_z*sb - uR*sb*b_z - ub*cb*b_z

    return ux_x, ux_y, ux_z, uy_x, uy_y, uy_z, uz_x, uz_y, uz_z
end

