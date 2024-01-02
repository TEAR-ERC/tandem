local ridgecrest = {}
ridgecrest.__index = ridgecrest

-- Directory of input files
fractal_dir = '/home/jyun/Tandem/perturb_stress'
stress_dir = '/home/jyun/Tandem/perturb_stress/seissol_outputs'

-- Frictional parameters
ridgecrest.a_b1 = 0.012
ridgecrest.a_b2 = -0.004
ridgecrest.a_b3 = 0.015
ridgecrest.a_b4 = 0.024
ridgecrest.b = 0.019

-- Shear stress [MPa]: negative for right-lateral
ridgecrest.tau1 = -10
ridgecrest.tau2 = -30
ridgecrest.tau3 = -22.5

-- Normal stress [MPa]: positive for compression
ridgecrest.sigma1 = 10
ridgecrest.sigma2 = 50

-- Depths where parameters vary [km]
ridgecrest.Wf = 24
ridgecrest.H = 12.0
ridgecrest.h = 5.0
ridgecrest.H2 = 2.0

-- Others
ridgecrest.Vp = 1e-9           -- Plate rate [m/s]
ridgecrest.rho0 = 2.670        -- Density []
ridgecrest.V0 = 1.0e-6         -- Reference slip rate [m/s]
ridgecrest.f0 = 0.6            -- Reference friction coefficient
ridgecrest.Dc = 0.002          -- Basevalue for Dc
ridgecrest.nu = 0.25           -- Poisson ratio


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

-- 2) Load from a 2D input file: e.g., stress perturbation
function readtxt_2D(file_name)
    local array = {}  -- Initialize the 2D array
    -- Open the file in read mode
    local file = io.open(file_name, "r")
    if file then
        for line in file:lines() do
            local inner_array = {}  -- Initialize an inner array to store elements of each line
            -- Split the line by a delimiter (assuming space in this case)
            for element in line:gmatch("%S+") do
                table.insert(inner_array, element)  -- Add elements to the inner array
            end
            table.insert(array, inner_array)  -- Add the inner array to the main 2D array
        end
        file:close()  -- Close the file
    else
        print("File not found or unable to open.")
    end
    return array
end

-- 3) Linear interpolation
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

-- 4) Read perturbation file and interpolate value at given 
function pert_at_y(delDat,dep,init_time,t,y,dt)
    local y0 = 0.0
    local ti = math.floor((t-init_time)/dt+0.5) + 1
    if ti <= #delDat then
        y0 = linear_interpolation(dep, delDat[ti],y)
    else
        y0 = linear_interpolation(dep, delDat[#delDat],y)
    end
    return y0
end

-- 5) Function to display the shape of a 2D array
function displayShape(array)
    local rows = #array  -- Get the number of rows
    if rows > 0 then
        local columns = #array[1]  -- Get the number of columns (assuming all rows have the same number of columns)
        print("(" .. rows .. ", " .. columns..")")
    else
        print("Array is empty")
    end
end

-------------------- Define your domain: below is the base values for all the parameters
function ridgecrest.new(params)
    local self = setmetatable({}, ridgecrest)
    self.model_n = params.model_n
    self.strike = params.strike
    self.fcoeff = params.fcoeff
    self.init_time = params.init_time
    self.dt = params.dt

    -- Define filenames
    local fname_fractal_sn = fractal_dir..'/fractal_snpre_06'
    local fname_fractal_ab = fractal_dir..'/fractal_ab_02'
    local fname_fractal_dc = fractal_dir..'/fractal_Dc_02'
    local fname_Pn = stress_dir..'/ssaf_'..self.model_n..'_Pn_pert_mu'..string.format("%02d",self.fcoeff*10)..'_'..tostring(self.strike)..'.dat'
    local fname_Ts = stress_dir..'/ssaf_'..self.model_n..'_Ts_pert_mu'..string.format("%02d",self.fcoeff*10)..'_'..tostring(self.strike)..'.dat'
    local fname_dep = stress_dir..'/ssaf_'..self.model_n..'_dep_stress_pert_mu'..string.format("%02d",self.fcoeff*10)..'_'..tostring(self.strike)..'.dat'
    local fname_final_stress = fractal_dir..'/final_stress_pert'

    -- Define your input data
    self.y_sn,self.fractal_sn = readtxt_1D(fname_fractal_sn,2)
    self.y_ab,self.fractal_ab = readtxt_1D(fname_fractal_ab,2)
    self.y_dc,self.fractal_dc = readtxt_1D(fname_fractal_dc,2)
    self.delPn = readtxt_2D(fname_Pn)
    self.delTs = readtxt_2D(fname_Ts)
    self.depinfo = readtxt_1D(fname_dep,1)


    return self
end

function ridgecrest:boundary(x, y, t)
    if x > 1.0 then
        return self.Vp/2.0 * t
    elseif x < -1.0 then
        return -self.Vp/2.0 * t
    else
        return self.Vp * t
    end
end

function ridgecrest:mu(x, y)
    return 20.0
end

function ridgecrest:eta(x, y)
    return math.sqrt(self:mu(x, y) * self.rho0) / 2.0
end

function ridgecrest:lam(x, y)
    return 2 * self.nu * self:mu(x,y) / (1 - 2 * self.nu)
end

function ridgecrest:Vinit(x, y)
    return 1.0e-9
end

function ridgecrest:L(x, y)
    local het_L = linear_interpolation(self.y_dc,self.fractal_dc, y)
    if y > 0 then
        het_L = self.Dc
    end
    return het_L
end

function ridgecrest:sn_pre(x, y)
    local het_sigma = linear_interpolation(self.y_sn,self.fractal_sn, y)
    if het_sigma == nil then
        het_sigma = self.sigma1
    end
    return het_sigma
end

function ridgecrest:delta_sn(x, y, t)
    local _del_sn = 0.0
    if t < self.init_time then
        -- print('t < init_time')
        _del_sn = 0.0
    else
        _del_sn = pert_at_y(self.delPn,self.depinfo,self.init_time,t,y,self.dt)
    end
    return _del_sn
end

function ridgecrest:tau_pre(x, y)
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

function ridgecrest:delta_tau(x, y, t)
    local _del_tau = 0.0
    if t < self.init_time then
        -- print('t < init_time')
        _del_tau = 0.0
    else
        _del_tau = -pert_at_y(self.delTs,self.depinfo,self.init_time,t,y,self.dt)
    end
    return _del_tau
end

function ridgecrest:ab(x, y)
    local het_ab = linear_interpolation(self.y_ab,self.fractal_ab, y)
    if y > 0 then
        het_ab = self.a_b1
    end
    return het_ab
end

function ridgecrest:a(x, y)
    local z = -y
    local _ab = self:ab(x,y)
    local _a = _ab + self.b
    return _a
end

return ridgecrest
