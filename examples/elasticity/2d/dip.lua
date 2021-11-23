local dip = -math.pi / 3.0
local mu = 1.0
local S = 1.0
local nu = 0.25
local lamda = 0.5

-- compute constants
local A_1, A_2, A_3, A_4, B_1, B_2, B_3, B_4
do
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
    B_3 = x16*(x1*x6 + x12*x13 - x12*x14 - x15*x2 + x15*x5 + x2 - x4*x6 - x5 + x7*x8 - x8*x9)
    B_4 = x16*(-x1*x8 - x12*x2 + x12*x5 - x13*x15 + x13 + x14*x15 - x14 + x4*x8 + x6*x7 - x6*x9)
end
do
    B_2 = -B_4
    B_1 = B_3*(1 - lamda)/(lamda + 1)
end
do
    local x0 = nu - 1
    local x1 = 1.0/x0
    local x2 = math.pi*((1.0/3.0)*lamda + 1.0/6.0)
    local x3 = (1.0/4.0)*S
    A_3 = x1*(B_3*x0 - x3*math.cos(x2))
    A_4 = x1*(B_4*x0 - x3*math.sin(x2))
end
do
    local x0 = 1.0/(lamda + 1)
    local x1 = 2*math.pi*lamda
    local x2 = math.cos(x1)
    local x3 = math.sin(x1)
    A_1 = x0*(-A_3*lamda + A_3*x2 - A_4*x3)
    A_2 = -x0*(A_3*x3 + A_4*lamda + A_4*x2)
end

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

function slip(x, y)
    local r, theta = polar(x, y)
    local ur = S * r^lamda / (2 * mu)
    local ut = 0
    return uxy(r, theta, ur, ut)
end

function sol_A(r, theta)
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
    return urA, utA
end

function jac_A(r, theta)
    local x0 = lamda - 1
    local x1 = theta*x0
    local x2 = math.sin(x1)
    local x3 = A_3*x2
    local x4 = math.cos(x1)
    local x5 = A_4*x4
    local x6 = x3 + x5
    local x7 = lamda + 1
    local x8 = theta*x7
    local x9 = math.sin(x8)
    local x10 = A_1*x9
    local x11 = math.cos(x8)
    local x12 = A_2*x11
    local x13 = (1.0/2.0)*r^lamda/mu
    local x14 = lamda*x13/r
    local x15 = 4*nu
    local x16 = A_3*x4
    local x17 = x0*x16
    local x18 = A_4*x2
    local x19 = x0*x18
    local x20 = x17 - x19
    local x21 = A_1*x11*x7
    local x22 = A_2*x7*x9
    local x23 = x15 - 4
    local x24 = x7^2
    local x25 = x0^2
    local urA_r = x14*(4*x6*(1 - nu) - x7*(x10 + x12 + x6))
    local urA_t = x13*(x20*(4 - x15) + (-lamda - 1)*(x20 + x21 - x22))
    local utA_r = x14*(-x17 + x19 - x21 + x22 + x23*(x16 - x18))
    local utA_t = x13*(x10*x24 + x12*x24 + x23*(-x0*x3 - x0*x5) + x25*x3 + x25*x5)
    return urA_r, urA_t, utA_r, utA_t
end

function sol_B(r, theta)
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
    return urB, utB
end

function jac_B(r, theta)
    local x0 = lamda - 1
    local x1 = theta*x0
    local x2 = math.sin(x1)
    local x3 = B_3*x2
    local x4 = math.cos(x1)
    local x5 = B_4*x4
    local x6 = x3 + x5
    local x7 = lamda + 1
    local x8 = theta*x7
    local x9 = math.sin(x8)
    local x10 = B_1*x9
    local x11 = math.cos(x8)
    local x12 = B_2*x11
    local x13 = (1.0/2.0)*r^lamda/mu
    local x14 = lamda*x13/r
    local x15 = 4*nu
    local x16 = B_3*x4
    local x17 = x0*x16
    local x18 = B_4*x2
    local x19 = x0*x18
    local x20 = x17 - x19
    local x21 = B_1*x11*x7
    local x22 = B_2*x7*x9
    local x23 = x15 - 4
    local x24 = x7^2
    local x25 = x0^2
    local urB_r = x14*(4*x6*(1 - nu) - x7*(x10 + x12 + x6))
    local urB_t = x13*(x20*(4 - x15) + (-lamda - 1)*(x20 + x21 - x22))
    local utB_r = x14*(-x17 + x19 - x21 + x22 + x23*(x16 - x18))
    local utB_t = x13*(x10*x24 + x12*x24 + x23*(-x0*x3 - x0*x5) + x25*x3 + x25*x5)
    return urB_r, urB_t, utB_r, utB_t
end

function solution(x, y)
    local r, theta = polar(x, y)
    local ur, ut
    if theta < dip then
        ur, ut = sol_A(r, theta)
    else
        ur, ut = sol_B(r, theta)
    end
    return uxy(r, theta, ur, ut)
end

function solution_jacobian(x, y)
    local r, theta = polar(x, y)
    local ur, ut, ur_r, ur_t, ut_r, ut_t
    if theta < dip then
        ur, ut = sol_A(r, theta)
        ur_r, ur_t, ut_r, ut_t = jac_A(r, theta)
    else
        ur, ut = sol_B(r, theta)
        ur_r, ur_t, ut_r, ut_t = jac_B(r, theta)
    end
    local c = math.cos(theta)
    local s = math.sin(theta)
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
