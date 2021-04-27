local dip = -math.pi / 3.0
local mu = 1.0
local S = 1.0
local nu = 0.25
local lamda = 0.5

function mat_mu(x, y)
    return mu
end

function mat_lam(x, y)
    return 2.0 * mu * nu / (1-2*nu)
end

function polar(x, y)
    local r = math.sqrt(x^2 + y^2)
    local theta = math.atan2(y, x)
    return r, theta
end

function uxy(r, theta, ur, ut)
    local ux = ur * math.cos(theta) - ut * math.sin(theta)
    local uy = ur * math.sin(theta) + ut * math.cos(theta)
    return ux, uy
end

function sol_A(x, y)
    local r, theta = polar(x, y)

    local x0 = math.pi*((1.0/3.0)*lamda + 1.0/3.0)
    local x1 = math.sin(x0)
    local x2 = 2*x1
    local x3 = math.pi*((7.0/3.0)*lamda + 1.0/3.0)
    local x4 = math.sin(x3)
    local x5 = 2*x4
    local x6 = 3*lamda
    local x7 = math.cos(x0)
    local x8 = math.sqrt(3)*lamda
    local x9 = math.cos(x3)
    local x10 = math.pi*lamda
    local x11 = (8.0/3.0)*x10
    local x12 = math.sin(x11)
    local x13 = 2*x7
    local x14 = 2*x9
    local x15 = math.cos(x11)
    local x16 = (1.0/32.0)*S/((nu - 1)*math.sin(x10)^2)
    local B_3 = x16*(x1*x6 + x12*x13 - x12*x14 - x15*x2 + x15*x5 + x2 - x4*x6 - x5 + x7*x8 - x8*x9)
    local B_4 = x16*(-x1*x8 - x12*x2 + x12*x5 - x13*x15 + x13 + x14*x15 - x14 + x4*x8 + x6*x7 - x6*x9)
    local x0 = nu - 1
    local x1 = 1.0/x0
    local x2 = math.pi*((1.0/3.0)*lamda + 1.0/6.0)
    local x3 = (1.0/4.0)*S
    local A_3 = x1*(B_3*x0 - x3*math.cos(x2))
    local A_4 = x1*(B_4*x0 - x3*math.sin(x2))
    local x0 = 1.0/(lamda + 1)
    local x1 = 2*math.pi*lamda
    local x2 = math.cos(x1)
    local x3 = math.sin(x1)
    local A_1 = x0*(-A_3*lamda + A_3*x2 - A_4*x3)
    local A_2 = -x0*(A_3*x3 + A_4*lamda + A_4*x2)
    local x0 = lamda - 1
    local x1 = theta*x0
    local x2 = math.sin(x1)
    local x3 = math.cos(x1)
    local x4 = A_3*x2 + A_4*x3
    local x5 = lamda + 1
    local x6 = theta*x5
    local x7 = math.sin(x6)
    local x8 = math.cos(x6)
    local x9 = (1.0/2.0)*r^lamda/mu
    local x10 = A_4*x2
    local x11 = A_3*x3
    local urA = x9*(4*x4*(1 - nu) - x5*(A_1*x7 + A_2*x8 + x4))
    local utA = x9*(-A_1*x5*x8 + A_2*x5*x7 + x0*x10 - x0*x11 + (4*nu - 4)*(-x10 + x11))

    local ux, uy = uxy(r, theta, urA, utA)
    return ux, uy
end

function sol_B(x, y)
    local r, theta = polar(x, y)

    local x0 = math.pi*((1.0/3.0)*lamda + 1.0/3.0)
    local x1 = math.sin(x0)
    local x2 = 2*x1
    local x3 = math.pi*((7.0/3.0)*lamda + 1.0/3.0)
    local x4 = math.sin(x3)
    local x5 = 2*x4
    local x6 = 3*lamda
    local x7 = math.cos(x0)
    local x8 = math.sqrt(3)*lamda
    local x9 = math.cos(x3)
    local x10 = math.pi*lamda
    local x11 = (8.0/3.0)*x10
    local x12 = math.sin(x11)
    local x13 = 2*x7
    local x14 = 2*x9
    local x15 = math.cos(x11)
    local x16 = (1.0/32.0)*S/((nu - 1)*math.sin(x10)^2)
    local B_3 = x16*(x1*x6 + x12*x13 - x12*x14 - x15*x2 + x15*x5 + x2 - x4*x6 - x5 + x7*x8 - x8*x9)
    local B_4 = x16*(-x1*x8 - x12*x2 + x12*x5 - x13*x15 + x13 + x14*x15 - x14 + x4*x8 + x6*x7 - x6*x9)
    local x0 = nu - 1
    local x1 = 1.0/x0
    local x2 = math.pi*((1.0/3.0)*lamda + 1.0/6.0)
    local x3 = (1.0/4.0)*S
    local A_3 = x1*(B_3*x0 - x3*math.cos(x2))
    local A_4 = x1*(B_4*x0 - x3*math.sin(x2))
    local B_2 = -B_4
    local B_1 = B_3*(1 - lamda)/(lamda + 1)
    local x0 = lamda - 1
    local x1 = theta*x0
    local x2 = math.sin(x1)
    local x3 = math.cos(x1)
    local x4 = B_3*x2 + B_4*x3
    local x5 = lamda + 1
    local x6 = theta*x5
    local x7 = math.sin(x6)
    local x8 = math.cos(x6)
    local x9 = (1.0/2.0)*r^lamda/mu
    local x10 = B_4*x2
    local x11 = B_3*x3
    local urB = x9*(4*x4*(1 - nu) - x5*(B_1*x7 + B_2*x8 + x4))
    local utB = x9*(-B_1*x5*x8 + B_2*x5*x7 + x0*x10 - x0*x11 + (4*nu - 4)*(-x10 + x11))


    local ux, uy = uxy(r, theta, urB, utB)
    return ux, uy
end

function solution(x, y)
    local r, theta = polar(x, y)
    if theta < dip then
        local ux, uy = sol_A(x, y)
        return ux, uy
    else
        local ux, uy = sol_B(x, y)
        return ux, uy
    end
end

function slip(x, y)
    local r, theta = polar(x, y)
    local ur = S * r^lamda / (2 * mu)
    local ut = 0
    local ux, uy = uxy(r, theta, ur, ut)
    return ux, uy
end
