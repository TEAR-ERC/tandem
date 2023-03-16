snprefile = io.open ('fractal_snpre','r')
lines = snprefile:lines()
local fault_y = {}
local sigma = {}
_y,_sigma = snprefile:read('*number', '*number')
if _y ~= nil then
    table.insert(fault_y,_y)
    table.insert(sigma,_sigma)
end

for line in lines do
    -- print(line)
    _y,_sigma = snprefile:read('*number', '*number')
    if _y ~= nil then
        table.insert(fault_y,_y)
        table.insert(sigma,_sigma)
    end
end
io.close(snprefile)

local function polynomial_interpolation(x, y, x0)
    local n = #x
    if x0 < x[1] or x0 > x[n] then
        return nil -- x0 is outside the range of x
    end
    local A = {}
    for i = 1, n do
        local row = {1}
        for j = 1, n-1 do
            table.insert(row, x[i]^j)
        end
        table.insert(A, row)
    end
    local b = {}
    for i = 1, n do
        table.insert(b, y[i])
    end
    local c = matrix.solve(matrix.new(A), matrix.new(b))
    local y0 = c[1]
    for i = 2, n do
        y0 = y0 + c[i] * x0^(i-1)
    end
    return y0
end

local y = -19.0
local het_sigma = polynomial_interpolation(fault_y, sigma, y)
