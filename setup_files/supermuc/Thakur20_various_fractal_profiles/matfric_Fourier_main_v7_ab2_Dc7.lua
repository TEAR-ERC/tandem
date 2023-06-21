local BP1 = {}

-- Frictional parameters
BP1.a_b1 = 0.012
BP1.a_b2 = -0.004
BP1.a_b3 = 0.015
BP1.a_b4 = 0.024
BP1.b = 0.019

-- Shear stress [MPa]: negative for right-lateral
BP1.tau1 = -10
BP1.tau2 = -30
BP1.tau3 = -22.5

-- Normal stress [MPa]: positive for compression
BP1.sigma1 = 10
BP1.sigma2 = 50

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
BP1.Dc = 0.002

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
y_sn,fractal_sn = read_fractal('/hppfs/work/pn49ha/di75weg/jeena-tandem/setup_files/supermuc/Thakur20_hetero_stress/fractal_snpre_07')
y_ab,fractal_ab = read_fractal('/hppfs/work/pn49ha/di75weg/jeena-tandem/setup_files/supermuc/Thakur20_various_fractal_profiles/fractal_ab_02')
y_dc,fractal_dc = read_fractal('/hppfs/work/pn49ha/di75weg/jeena-tandem/setup_files/supermuc/Thakur20_various_fractal_profiles/fractal_Dc_07')

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
    return 20.0
end

function BP1:eta(x, y)
    return math.sqrt(self:mu(x, y) * self.rho0) / 2.0
end

function BP1:L(x, y)
    local het_L = linear_interpolation(y_dc, fractal_dc, y)
    if y > 0 then
        het_L = self.Dc
    end
    file = io.open ('/hppfs/work/pn49ha/di75weg/jeena-tandem/setup_files/supermuc/Thakur20_various_fractal_profiles/dc_profile_v7_ab2_Dc7','a')
    io.output(file)
    io.write(y,'\t',het_L,'\n')
    io.close(file)
    return het_L
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
    file = io.open ('/hppfs/work/pn49ha/di75weg/jeena-tandem/setup_files/supermuc/Thakur20_various_fractal_profiles/ab_profile_v7_ab2_Dc7','a')
    io.output(file)
    io.write(y,'\t',_a,'\t',self.b,'\n')
    io.close(file)
    return _a
end

function BP1:tau_pre(x, y)
    local z = -y
    local _tau1 = self.tau2 + (self.tau2 - self.tau1) * (z - self.H2) / self.H2
    local _tau2 = self.tau2 + (self.tau3 - self.tau2) * (z - self.H) / self.h
    local _tau = self.tau3
    local _sn = self:sn_pre(x,y)

    if z < self.H2 then
        _tau = _tau1
    elseif z < self.H then
        _tau = self.tau2
    elseif z < self.H + self.h then
        _tau = _tau2
    end
    return _tau
end

bp1 = BP1:new()

bp1_sym = BP1:new()
function bp1_sym:boundary(x, y, t)
    return self.Vp/2.0 * t
end

