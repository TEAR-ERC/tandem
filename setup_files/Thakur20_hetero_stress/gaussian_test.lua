-- A tool for generating Gaussian random number
-- math.random(); math.random(); math.random()
math.randomseed(2023)
local function gaussian(mean, std)
    local variance = std^2
    local randn =  math.sqrt(-2 * variance * math.log(math.random())) * math.cos(2 * math.pi * math.random()) + mean
    return randn
end

-- file = io.open ('gaussian_distribution.txt','w')
-- io.output(file)
-- math.random()
-- for i=1,1000 do
--     a = gaussian(0,5)
--     io.write(a,'\n')
-- end
-- io.close(file)

local poisson = 0.25      -- Poisson ratio
local alpha = 1e-3        -- Roughness
local lambda_min = 200    -- Minimum fractal surface wavelength [m]
local mu = 32

local function het_sigma(x,y)
    local Gprime = mu/(1 - poisson)
    local std_sigma = 2*math.pi*math.pi*alpha*Gprime/lambda_min
    local pert = gaussian(0,std_sigma)
    return pert*1.0e+3
end

local function rough_surface()
    local a = np.zeros(N, dtype=complex)
    local beta = 2 * H + 1.0
    for i in range(0, Nmax + 1) do
        randPhase = np.random.rand() * np.pi * 2.0
        if i == 0 then
            fac = 0.0
        elseif (lambdaMax * i / L) ** 2 < 1.0 then
            -- remove lambda>lambdaMaxS
            fac = 0.0
        elseif (lambdaMin * i / L) ** 2 > 1.0 then
            -- remove lambda<lambdaMin
            fac = 0.0
        else
            fac = np.power(i, -beta)

        a[i] = fac * np.exp(randPhase * 1.0j)
        if i != 0:
            a[N - i] = np.conjugate(a[i])
        end
    end
    a = a * N
    h = np.real(np.fft.ifft(a))

    dx = 1.0 * L / (N - 1)
    x = np.arange(0, L + dx, dx)
    nx = x.shape[0]
    h = h[0:nx]
end

mshfile = io.open ('coord.txt','r')
lines = mshfile:lines()
x,y = mshfile:read('*number', '*number')
print('x =',x,', y =',y)
pert = het_sigma(x,y)

file = io.open ('sigma_perturb.txt','w')
file:write(pert,'\n')

for line in lines do
    -- print(line)
    x,y = mshfile:read('*number', '*number')
    if x ~= nil then
        print('x =',x,', y =',y)
        pert = het_sigma(x,y)
        file:write(pert,'\n')
    end
end
io.close(mshfile)
io.close(file)

