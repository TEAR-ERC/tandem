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
    local ti = 1.0e+6
    if init_time > 0 then
        ti = math.floor((t-init_time)/dt+0.5) + 1
    end
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


-------------------- Define your input data
local fname_Pn = '/home/jyun/Tandem/perturb_stress/seissol_outputs/ssaf_vert_slow_Pn_pert_mu04_340.dat'
local fname_Ts = '/home/jyun/Tandem/perturb_stress/seissol_outputs/ssaf_vert_slow_Ts_pert_mu04_340.dat'
local fname_dep = '/home/jyun/Tandem/perturb_stress/seissol_outputs/ssaf_vert_slow_dep_stress_pert_mu04_340.dat'
local fname_ab = '/home/jyun/Tandem/perturb_stress/fractal_ab_02'

-- local delPn = readtxt_2D(fname_Pn)
local delTs = readtxt_2D(fname_Pn)
-- local delPn = static_dCFS(fname_Pn)
-- print(#delPn)
-- print(#delPn[#delPn])
-- print(#delPn)
-- print(#delPn[1501])
-- displayShape(delPn)
-- local delTs = readtxt_2D(fname_Ts)
-- displayShape(delTs)
local dep = readtxt_1D(fname_dep,1)
-- print("("..#dep..",)")
-- local y,ab = readtxt_1D(fname_ab,2)
-- print("("..#y..",)")
-- print("("..#ab..",)")

-- Interpolation test
-- local x0 = -25
-- y0 = linear_interpolation(dep, delPn[501], x0)
-- print(y0)
-- Works well

-- function routine_read_pert(delPn,t,x0)
--     local ti = math.floor((t-init_time)/0.01+0.5) + 1
--     print('Step index'..ti)
--     y0 = linear_interpolation(dep, delPn[ti], x0)
--     return y0
-- end

local init_time = 2.712094562139096069e+10
local y0 = -4.0
local t = 2.712094562140096e+10
-- local y = pert_at_y(delPn,dep,init_time,t,y0,0.01)
local y = pert_at_y(delTs,dep,init_time,t,y0,0.01)
-- local y = pert_at_y(delPn,dep,-1,t,y0,0.01)
print('Interpolated value at x0 = '..y0..' km at time = '..t..' s is '..y)