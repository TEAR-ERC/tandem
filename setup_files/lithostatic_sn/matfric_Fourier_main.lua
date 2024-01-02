local lithostatic = {}
lithostatic.__index = lithostatic

-- Directory of input files
fractal_dir = '/home/jyun/Tandem/lithostatic_sn'

-- Frictional parameters
lithostatic.a_b1 = 0.012
lithostatic.a_b2 = -0.004
lithostatic.a_b3 = 0.015
lithostatic.a_b4 = 0.024
lithostatic.b = 0.019

-- Depths where parameters vary [km]
lithostatic.Wf = 24
lithostatic.H = 12.0
lithostatic.h = 5.0
lithostatic.H2 = 2.0

-- Base values
lithostatic.sigma = 10          -- Basevalue for normal stress; positive compressive
lithostatic.tau = 4             -- Basevalue for shear stress; negative right-lateral
lithostatic.Dc = 0.002          -- Basevalue for Dc

-- Others
lithostatic.Vp = 1e-9           -- Plate rate [m/s]
lithostatic.rho0 = 2.670        -- Density []
lithostatic.V0 = 1.0e-6         -- Reference slip rate [m/s]
lithostatic.f0 = 0.6            -- Reference friction coefficient
lithostatic.nu = 0.25           -- Poisson ratio


-------------------- Define useful functions 
-- 1) Load from a 1D input file: e.g., fractal parameters
function readtxt_1D(fname,colnum)
    local file_name = io.open (fname,'r')
    local lines = file_name:lines()
    local var1 = {}
    if colnum == 1 then
        _var1 = file_name:read('*number')
        if _var1 ~= nil then
            table.insert(var1,_var1)
        end
        for line in lines do
            _var1 = file_name:read('*number')
            if _var1 ~= nil then
                table.insert(var1,_var1)
            end
        end
        io.close(file_name)
        return var1
    elseif colnum == 2 then
        local var2 = {}
        local _var1,_var2 = file_name:read('*number', '*number')
        if _var1 ~= nil then
            table.insert(var1,_var1)
            table.insert(var2,_var2)
        end
        for line in lines do
            local _var1,_var2 = file_name:read('*number', '*number')
            if _var1 ~= nil then
                table.insert(var1,_var1)
                table.insert(var2,_var2)
            end
        end
        io.close(file_name)
        return var1,var2
    end
end

-- 2) Linear interpolation
function linear_interpolation(x, y, x0)
    local n = #x
    local y0 = nil
    if x0 < x[1] or x0 > x[n] then -- x0 is out of range, take the last value
        if x0 < x[1] then
            y0 = y[1]
        elseif x0 > x[n] then
            y0 = y[n]
        end
    end
    for i = 1, n-1 do
        if x0 >= x[i] and x0 <= x[i+1] then
            local slope = (y[i+1] - y[i]) / (x[i+1] - x[i])
            y0 = y[i] + slope * (x0 - x[i])
        end
    end
    return y0
end

-- 3)
function get_fname(frac_type,raw_model_n)
    local fname_fractal = fractal_dir..'/'..frac_type..'_0'..raw_model_n
    if raw_model_n >= 10 then
        fname_fractal = fractal_dir..'/fractal_snpre_'..raw_model_n
    end
    return fname_fractal
end
-------------------- Define your domain: below is the base values for all the parameters
function lithostatic.new(params)
    local self = setmetatable({}, lithostatic)
    self.frac_sn_model_n = params.sn
    self.frac_ab_model_n = params.ab
    self.frac_dc_model_n = params.dc

    -- Define filenames
    if self.frac_sn_model_n > 0 then 
        local fname_fractal_sn = get_fname('fractal_litho_snpre',self.frac_sn_model_n) 
        self.y_sn,self.fractal_sn = readtxt_1D(fname_fractal_sn,2)
    end
    if self.frac_ab_model_n > 0 then 
        local fname_fractal_ab = get_fname('fractal_ab',self.frac_ab_model_n) 
        self.y_ab,self.fractal_ab = readtxt_1D(fname_fractal_ab,2)
    end
    if self.frac_dc_model_n > 0 then 
        local fname_fractal_dc = get_fname('fractal_Dc',self.frac_dc_model_n) 
        self.y_dc,self.fractal_dc = readtxt_1D(fname_fractal_dc,2)
    end

    return self
end

function lithostatic:boundary(x, y, t)
    if x > 1.0 then
        return self.Vp/2.0 * t
    elseif x < -1.0 then
        return -self.Vp/2.0 * t
    else
        return self.Vp * t
    end
end

function lithostatic:mu(x, y)
    return 20.0
end

function lithostatic:eta(x, y)
    return math.sqrt(self:mu(x, y) * self.rho0) / 2.0
end

function lithostatic:lam(x, y)
    return 2 * self.nu * self:mu(x,y) / (1 - 2 * self.nu)
end

function lithostatic:Vinit(x, y)
    return 1.0e-9
end

function lithostatic:L(x, y)
    local Dc = self.Dc
    if self.frac_dc_model_n > 0 then 
        local het_Dc = linear_interpolation(self.y_dc,self.fractal_dc, y)
        if y <= 0 then
            Dc = het_Dc
        end
        file = io.open (fractal_dir..'/dc_profile_'..self.model_n,'a')
        io.output(file)
        io.write(y,'\t',Dc,'\n')
        io.close(file)
    end
    return Dc
end

function lithostatic:sn_pre(x, y)
    local sigma = self.sigma
    if self.frac_sn_model_n > 0 then 
        local het_sigma = linear_interpolation(self.y_sn,self.fractal_sn, y)
        if het_sigma ~= nil then
            sigma = het_sigma
        end
    end
    return sigma
end

function lithostatic:tau_pre(x, y)
    local z = -y
    local sigma_base = self:sn_pre(x, y)
    local _tau = -sigma_base*self.f0 - self.tau
    return _tau
end

function lithostatic:ab(x, y)
    local ab = self.a_b
    if self.frac_ab_model_n > 0 then 
        local het_ab = linear_interpolation(self.y_ab,self.fractal_ab, y)
        if het_ab ~= nil then
            ab = het_ab
        end
    end
    return ab
end

function lithostatic:a(x, y)
    local z = -y
    local _ab = self:ab(x,y)
    local _a = _ab + self.b
    if self.frac_ab_model_n > 0 then 
        file = io.open (fractal_dir..'/ab_profile_'..self.model_n,'a')
        io.output(file)
        io.write(y,'\t',_a,'\t',self.b,'\n')
        io.close(file)
    end
    return _a
end

v3ab2Dc2 = lithostatic.new{sn=3,ab=2,dc=2}