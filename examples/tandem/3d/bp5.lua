local BP5 = {}
BP5.__index = BP5

BP5.b = 0.03
BP5.V0 = 1.0e-6
BP5.f0 = 0.6
BP5.Vp = 1e-9
BP5.Vzero = 1e-20

BP5.a0 = 0.004
BP5.amax = 0.04
BP5.h_s = 2.0
BP5.h_t = 2.0
BP5.H = 12.0
BP5.w = 12.0
BP5.l = 60
BP5.rho0 = 2.670
BP5.cs = 3.464
BP5.nu = 0.25

function BP5.new(params)
    local self = setmetatable({}, BP5)
    self.eps = params.eps
    return self
end

function BP5:boundary(x, y, z, t)
    local Vh = self.Vp * t
    if y < 0.001 then
        Vh = 0
    end
    return Vh, 0, 0
end

function BP5:mu(x, y, z)
    return self.cs^2 * self.rho0
end

function BP5:lam(x, y, z)
    return 2 * self.nu * self:mu(x,y) / (1 - 2 * self.nu)
end

function BP5:eta(x, y, z)
    return self.cs * self.rho0 / 2.0
end

function BP5:in_nucleation(x, y, z)
    local d = -z
    local s = x
    local eps = self.eps
    if self.h_s + self.h_t <= d+eps and d-eps <= self.h_s + self.h_t + self.H and -self.l/2.0 <= s+eps and s-eps <= -self.l/2.0 + self.w then
        return true
    end
    return false
end

function BP5:Lini(x, y, z)
    if self:in_nucleation(x, y, z) then
        return 0.13
    end
    return 0.14
end





function BP5:Sinit(x, y, z)
    return 0.0, 0.0
end

function BP5:Vinit(x, y, z)
    if self:in_nucleation(x, y, z) then
        return self.Vzero, 0.01
    end
    return self.Vzero, self.Vp
end


BP5.transition_poly_a = {
    {0.50000000000000000},
    {0.75000000000000000, -0.25000000000000000},
    {0.93750000000000000, -0.62500000000000000, 0.18750000000000000},
    {1.09375000000000000, -1.09375000000000000, 0.65625000000000000, -0.15625000000000000},
    {1.23046875000000000, -1.64062500000000000, 1.47656250000000000, -0.70312500000000000, 0.13671875000000000},
    {1.35351562500000000, -2.25585937500000000, 2.70703125000000000, -1.93359375000000000, 0.75195312500000000, -0.12304687500000000},
    {1.46630859375000000, -2.93261718750000000, 4.39892578125000000, -4.18945312500000000, 2.44384765625000000, -0.79980468750000000, 0.11279296875000000},
    {1.57104492187500000, -3.66577148437500000, 6.59838867187500000, -7.85522460937500000, 6.10961914062500000, -2.99926757812500000, 0.84594726562500000, -0.10473632812500000},
    {1.66923522949218750, -4.45129394531250000, 9.34771728515625000, -13.3538818359375000, 12.9829406738281250, -8.49792480468750000, 3.59527587890625000, -0.89025878906250000, 0.09819030761718750}
}

-- p: polynomial order
-- x in [-1,+1]: local coordinate
-- output: P_n(x) in [0,1]
function BP5:transition_poly(p, x)
    --    c: level of continuity, e.g. 1: zeroth and first derivatives are continuous.
    local c = (p-1)/2 -- this should round down to an integer, check!
    -- if c + 1 > 9 (or p > 32), tell Casper to generate a longer table. maybe raise an error here.
    local a = self.transition_poly_a[c+1]
    local f = 0.5
    for i = 0, c do
        f = f + a[i+1] * x^(1+2*i)
    end
    return f
end


-- H: size of patch (minus the half of the smooth boundary)
-- h: size of the smooth boundary
-- x, y: fault coordinates
-- output: phase field in [0,1]
function BP5:smooth_square_patch(p, x, y, Hx, Hy, hx, hy)
    local px = {0, self:transition_poly(p, (x + 0.5*Hx)/(0.5*hx)), 1, 1-self:transition_poly(p, (x - 0.5*Hx)/(0.5*hx))}
    local py = {0, self:transition_poly(p, (y + 0.5*Hy)/(0.5*hy)), 1, 1-self:transition_poly(p, (y - 0.5*Hy)/(0.5*hy))}
    local ix = 1
    local iy = 1
    -- be sure to mesh the regions described below explicitly!
    if - 0.5*Hx - 0.5*hx <= x and x< - 0.5*Hx + 0.5*hx then ix = 2 end
    if - 0.5*Hx+ 0.5*hx <= x and x<   0.5*Hx - 0.5*hx then ix = 3 end
    if   0.5*Hx - 0.5*hx <= x and x<   0.5*Hx + 0.5*hx then ix = 4 end
    if - 0.5*Hy - 0.5*hy <= y and y< - 0.5*Hy + 0.5*hy then iy = 2 end
    if - 0.5*Hy + 0.5*hy <= y and y<   0.5*Hy - 0.5*hy then iy = 3 end
    if   0.5*Hy - 0.5*hy <= y and y<   0.5*Hy + 0.5*hy then iy = 4 end
    return px[ix] * py[iy]
end

-- alternative function to intialize using a smooth_square_patch
-- debbugged from Casper initial code
function BP5:a_alternative(x, y, z)
    local d = -z
    local s = math.abs(x)
    local z_center = - (0.5*self.H + self.h_t + self.h_s)
    local phi = self:smooth_square_patch(3, x, z-z_center, self.l+self.h_t, self.H+self.h_s, self.h_t, self.h_s)
    return (1-phi)*self.amax + phi*self.a0
end

function BP5:L(x, y, z)
    local x_center = - (0.5*self.l - 0.5*self.w)
    local z_center = - (0.5*self.H + self.h_t + self.h_s)
    local phi = self:smooth_square_patch(3, x-x_center, z-z_center, self.w+self.h_t, self.H+self.h_t, self.h_t, self.h_t)
    return (1-phi)*0.14 + phi*0.13
end

function BP5:a(x, y, z)
    local d = -z
    local s = math.abs(x)
    if self.h_s + self.h_t <= d and d <= self.h_s + self.h_t + self.H and s <= self.l/2 then
        return self.a0
    elseif d <= self.h_s or self.h_s + 2*self.h_t + self.H <= d or self.l/2 + self.h_t <= s then
        return self.amax
    else
        local r = math.max(math.abs(d-self.h_s-self.h_t-self.H/2)-self.H/2, s-self.l/2)/self.h_t
        return self.a0 + r*(self.amax-self.a0)
    end
end

function BP5:sn_pre(x, y, z)
    return 25.0
end

function BP5:tau_pre(x, y, z)
    local Vi1, Vi2 = self:Vinit(x, y, z)
    local Vi = math.sqrt(Vi1^2 + Vi2^2)
    local sn = self:sn_pre(x, y, z)
    local ax = self:a(x, y, z)
    local e = math.exp((self.f0 + self.b * math.log(self.V0 / self.Vp)) / ax)
    local tau0 = sn * ax * math.asinh((Vi2 / (2.0 * self.V0)) * e)
                 + self:eta(x, y, z) * Vi2
    return -tau0 * Vi1 / Vi, -tau0 * Vi2 / Vi
end

bp5_exact = BP5.new({eps=0.0})
bp5_outside = BP5.new({eps=1e-3})
bp5_inside = BP5.new({eps=-1e-3})

