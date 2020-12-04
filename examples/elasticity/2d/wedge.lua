local S = 1.0
local mu = 1.0
local nu = 0.25

function mat_mu(x, y)
    return mu
end

function mat_lam(x, y)
    return 2.0 * mu * nu / (1-2*nu)
end

function solution(x, y)
    local r = math.sqrt(x^2 + y^2)
    local theta = math.atan2(y, x)

    local A_1 = (1.0/3.0)*math.sqrt(3)*S*mu*(4*nu - 5)/(-16*nu^2 + 20*nu + 5)
    local A_2 = S*mu*(7 - 4*nu)/(-16*nu^2 + 20*nu + 5)
    local A_3 = math.sqrt(3)*S*mu*(5 - 4*nu)/(-16*nu^2 + 20*nu + 5)
    local A_4 = S*mu*(4*nu - 7)/(-16*nu^2 + 20*nu + 5)
    local ur = (1.0/2.0)*r^2*(-3*A_1*math.sin(3*theta) - 3*A_2*math.cos(3*theta) - 4*A_3*nu*math.sin(theta) + A_3*math.sin(theta) - 4*A_4*nu*math.cos(theta) + A_4*math.cos(theta))/mu
    local ut = (1.0/2.0)*r^2*(-3*A_1*math.cos(3*theta) + 3*A_2*math.sin(3*theta) - A_3*math.cos(theta) + A_4*math.sin(theta) + (4*nu - 4)*(A_3*math.cos(theta) - A_4*math.sin(theta)))/mu

    local ux = ur * math.cos(theta) - ut * math.sin(theta)
    local uy = ur * math.sin(theta) + ut * math.cos(theta)
    return ux, uy
end
