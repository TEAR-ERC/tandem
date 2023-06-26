local BP1 = {}

-- Frictional parameters
BP1.a_b1 = 0.012
BP1.a_b2 = -0.004
BP1.a_b3 = 0.015
BP1.a_b4 = 0.024
BP1.b = 0.019

-- Base normal stress profile [MPa]
BP1.surf = 10
BP1.sigma_grad = 15

-- Depths where parameters vary [km]
BP1.Wf = 24
BP1.H = 12.0
BP1.h = 5.0
BP1.H2 = 2.0

-- Others
BP1.Vp = 1e-9           -- Plate rate [m/s]
BP1.rho0 = 2.670        -- Density []
BP1.V0 = 1.0e-6         -- Reference slip rate [m/s]
BP1.f0 = 0.6            -- Reference friction coefficient
BP1.Dc = 0.004

-------------------- Define a function that reads in your input fractal profile
function read_fractal(fname)
    local fractal_file = io.open (fname,'r')
    local lines = fractal_file:lines()
    local fault_y = {}
    local var = {}
    local _y,_var = fractal_file:read('*number', '*number')
    if _y ~= nil then
        table.insert(fault_y,_y)
        table.insert(var,_var)
    end

    for line in lines do
        _y,_var = fractal_file:read('*number', '*number')
        if _y ~= nil then
            table.insert(fault_y,_y)
            table.insert(var,_var)
        end
    end
    io.close(fractal_file)
    return fault_y,var
end

-------------------- Define your input data
y_sn,fractal_sn = read_fractal('/home/jyun/Tandem/lithostatic_sn/fractal_litho_snpre_02')
y_ab,fractal_ab = read_fractal('/home/jyun/Tandem/Thakur20_various_fractal_profiles/fractal_ab_02')

-------------------- Define linear interpolation function
function linear_interpolation(x, y, x0)
    local n = #x
    if x0 < x[1] or x0 > x[n] then
        return nil -- x0 is outside the range of x
    end
    for i = 1, n-1 do
        if x0 >= x[i] and x0 <= x[i+1] then
            local slope = (y[i+1] - y[i]) / (x[i+1] - x[i])
            return y[i] + slope * (x0 - x[i])
        end
    end
end

-------------------- Define your domain: below is the base values for all the parameters
function BP1:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function BP1:boundary(x, y, t)
    if x > 1.0 then
        return self.Vp/2.0 * t
    elseif x < -1.0 then
        return -self.Vp/2.0 * t
    else
        return self.Vp * t
    end
end

function BP1:mu(x, y)
    return 15.0
end

function BP1:eta(x, y)
    return math.sqrt(self:mu(x, y) * self.rho0) / 2.0
end

function BP1:L(x, y)
    return 0.004
end

function BP1:sn_pre(x, y)
    local het_sigma = linear_interpolation(y_sn, fractal_sn, y)
    if het_sigma == nil then
        het_sigma = self.sigma1
    end
    return het_sigma
end

function BP1:Vinit(x, y)
    return 1.0e-9
end

function BP1:ab(x, y)
    local het_ab = linear_interpolation(y_ab, fractal_ab, y)
    if y > 0 then
        het_ab = self.a_b1
    end
    return het_ab
end

function BP1:a(x, y)
    local z = -y
    local _ab = self:ab(x,y)
    local _a = _ab + self.b
    file = io.open ('/home/jyun/Tandem/lithostatic_sn/ab_profile_v2_ab2','a')
    io.output(file)
    io.write(y,'\t',_a,'\t',self.b,'\n')
    io.close(file)
    return _a
end

function BP1:tau_pre(x, y)
    -- Shear stress [MPa]: negative for right-lateral
    local z = -y
    local sigma_base = self.sigma_grad*z
    return -sigma_base*self.f0 - self.surf
end

bp1 = BP1:new()

bp1_sym = BP1:new()
function bp1_sym:boundary(x, y, t)
    return self.Vp/2.0 * t
end

