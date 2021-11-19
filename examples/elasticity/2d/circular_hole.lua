local S = 1.0
local a = 0.2
local mu0 = 1.0
local nu = 0.25

function mu(x, y)
    return mu0
end

function lam(x, y)
    return 2 * mu0 * nu / (1 - 2*nu)
end

function force(x, y)
    return 0, 0
end

function polar(x, y)
    local r = math.sqrt(x^2 + y^2)
    local t = math.atan2(y, x)
    return r, t
end

function u_polar(r, t)
    local ur = (1.0/2.0)*S*(-a^4 - 4*a^2*r^2*(nu - 1) + r^4)*math.sin(2*t)/(mu0*r^3)
    local ut = (1.0/2.0)*S*(a^4 - 4*a^2*nu*r^2 + 2*a^2*r^2 + r^4)*math.cos(2*t)/(mu0*r^3)
    return ur, ut 
end

function solution(x, y)
    local r, t = polar(x, y)
    local ur, ut = u_polar(r, t)
    return ur * math.cos(t) - ut * math.sin(t), ur * math.sin(t) + ut * math.cos(t)
end

function solution_jacobian(x, y)
    local r, t = polar(x, y)
    local c = math.cos(t)
    local s = math.sin(t)
    local ur, ut = u_polar(r, t)
    local ur_r = (1.0/2.0)*S*(3*a^4 + 4*a^2*nu*r^2 - 4*a^2*r^2 + r^4)*math.sin(2*t)/(mu0*r^4)
    local ur_t = -S*(a^4 + 4*a^2*r^2*(nu - 1) - r^4)*math.cos(2*t)/(mu0*r^3)
    local ut_r = (1.0/2.0)*S*(-3*a^4 + 4*a^2*nu*r^2 - 2*a^2*r^2 + r^4)*math.cos(2*t)/(mu0*r^4)
    local ut_t = S*(-a^4 + 4*a^2*nu*r^2 - 2*a^2*r^2 - r^4)*math.sin(2*t)/(mu0*r^3)
    local r_x = c
    local r_y = s
    local t_x = -s / r
    local t_y = c / r
    local ur_x = ur_r * r_x + ur_t * t_x
    local ur_y = ur_r * r_y + ur_t * t_y
    local ut_x = ut_r * r_x + ut_t * t_x
    local ut_y = ut_r * r_y + ut_t * t_y

    local ux_x = c * ur_x - s * ut_x - t_x * (s * ur + c * ut)
    local ux_y = c * ur_y - s * ut_y - t_y * (s * ur + c * ut)
    local uy_x = s * ur_x + c * ut_x + t_x * (c * ur - s * ut)
    local uy_y = s * ur_y + c * ut_y + t_y * (c * ur - s * ut)

    return ux_x, ux_y, uy_x, uy_y
end

