local mtmod = {}
mtmod.__index = mtmod

mtmod.cs = 3.464
mtmod.nu = 0.25
mtmod.b = 0.015
mtmod.V0 = 1.0e-6
mtmod.f0 = 0.6
mtmod.dip = 30 * math.pi / 180.0
mtmod.Vp = 1e-9
mtmod.rho0 = 2.670
mtmod.H = 60.0 * math.sin(mtmod.dip)
mtmod.h = 20.0 * math.sin(mtmod.dip)
mtmod.H2 = 80.0 * math.sin(mtmod.dip)


function mtmod.new(params)
    local self = setmetatable({}, mtmod)
    self.sigma_vs = params.sigma_vs
    self.sigma_asperity = params.sigma_asperity
    self.sigma_shallow = params.sigma_shallow
    self.avs = params.avs
    self.avw_shallow = params.avw_shallow
    self.avw_asperity = params.avw_asperity
    self.Dc = params.Dc
    self.Dc_asperity = params.Dc_asperity
    self.pathstr = params.branch_n
    self.muver = params.muver
    if self.muver == 'patch' then
        self.cs_high = params.cs_high
        self.cs_low = params.cs_low
        self.mu_sed = params.mu_sed
        self.sed_xlim = params.sed_xlim 
        self.sed_ylim = params.sed_ylim 
    end
    return self
end

function mtmod:boundary(x, y, t)
    local Vh = self.Vp * t
    local dist = x + y / math.tan(self.dip)
    if dist > 1 then
        Vh = -Vh / 2.0
    elseif dist < -1 then
        Vh = Vh / 2.0
    end
    --return Vh * math.cos(self.dip), -Vh * math.sin(self.dip)
    return Vh, 0
end

function mtmod:mu(x, y)
    if self.muver == 'hom' then
        -- Homogeneous ver.
        local base_mu = self.cs^2 * self.rho0
        _mu = base_mu
    elseif self.muver == 'grad' then
        -- Gradient ver.
        local z = -y
        _mu = math.sqrt(z*60)+2.5
        file = io.open ('/home/jyun/Tandem/mtmod_team3/mu_profile_'..self.pathstr,'a')
        io.output(file)
        io.write(x,'\t',y,'\t',_mu,'\n')
        io.close(file)
    elseif self.muver == 'patch' then
        -- Gradient ver.
        local z = -y
        local mu_high = self.cs_high^2*self.rho0
        local mu_low = self.cs_low^2*self.rho0
        _mu = mu_high
        if x >= -z / math.tan(math.pi - self.dip) then
            if x <= self.sed_xlim and y <= self.sed_ylim then
                _mu = self.mu_sed
            else
                _mu = mu_low
            end
        end
        file = io.open ('/home/jyun/Tandem/mtmod_team3/mu_profile_'..self.pathstr,'a')
        io.output(file)
        io.write(x,'\t',y,'\t',_mu,'\n')
        io.close(file)
    end

    return _mu
end

function mtmod:lam(x, y)
    return 2 * self.nu * self:mu(x,y) / (1 - 2 * self.nu)
end

function mtmod:eta(x, y)
    return self.cs * self.rho0 / 2.0
end

function mtmod:L(x, y)
    local z = -y
    -- Linear Dc
    -- local _dc1 = self.Dc_asperity + (self.Dc_asperity - self.Dc) * (z - self.h) / self.h
    -- local _dc2 = self.Dc + (self.Dc - self.Dc_asperity) * (z - self.H2) / (self.H2-self.H)
    -- _Dc = self.Dc
    -- if z < self.h then
    --     _Dc = _dc1
    -- elseif z < self.H then
    --     _Dc = self.Dc_asperity
    -- elseif z < self.H2 then
    --     _Dc = _dc2
    -- else
    --     _Dc = self.Dc
    -- end

    -- Discrete Dc
    _Dc = self.Dc
    if z < self.h then
        _Dc = self.Dc
    elseif z < self.H then
        _Dc = self.Dc_asperity
    else
        _Dc = self.Dc
    end

    file = io.open ('/home/jyun/Tandem/mtmod_team3/dc_profile_'..self.pathstr,'a')
    io.output(file)
    io.write(x,'\t',y,'\t',_Dc,'\n')
    io.close(file)

    return _Dc
end

function mtmod:Sinit(x, y)
    return 0.0
end

function mtmod:Vinit(x, y)
    return self.Vp
end

function mtmod:a(x, y)
    local z = math.abs(y)
    -- Linear a
    -- local _a1 = self.avw_asperity + (self.avw_asperity - self.avw_shallow) * (z - self.h) / self.h
    -- local _a2 = self.avs + (self.avs - self.avw_asperity) * (z - self.H2) / (self.H2-self.H)
    -- _a = self.avs
    -- if z < self.h then
    --     _a = _a1
    -- elseif z < self.H then
    --     _a = self.avw_asperity
    -- elseif z < self.H2 then
    --     _a = _a2
    -- else
    --     _a = self.avs
    -- end

    -- Discrete a
    _a = self.avs
    if z < self.h then
        _a = self.avw_shallow
    elseif z < self.H then
        _a = self.avw_asperity
    else
        _a = self.avs
    end
    
    file = io.open ('/home/jyun/Tandem/mtmod_team3/ab_profile_'..self.pathstr,'a')
    io.output(file)
    io.write(x,'\t',y,'\t',_a,'\t',self.b,'\n')
    io.close(file)

    return _a
end

function mtmod:sn_pre(x, y)
    -- positive in compression
    local z = math.abs(y)
    -- Linear Sn
    -- local _sn1 = self.sigma_asperity + (self.sigma_asperity - self.sigma_shallow) * (z - self.h) / self.h
    -- _sn = self.sigma_vs
    -- if z < self.h then
    --     -- _sn = self.sigma_shallow
    --     _sn = _sn1 
    -- elseif z < self.H then
    --     _sn = self.sigma_asperity
    -- else
    --     _sn = self.sigma_vs
    -- end

    -- Discrete Sn
    _sn = self.sigma_vs
    if z < self.h then
        _sn = self.sigma_shallow
    elseif z < self.H then
        _sn = self.sigma_asperity
    else
        _sn = self.sigma_vs
    end

    return _sn
end

function mtmod:tau_pre(x, y)
    -- Tau ss
    -- local z = math.abs(y)
    -- local sn = self:sn_pre(x, y)
    -- _tau = -(self.f0 * sn + math.log(self.Vp / math.cos(self.dip) / self.V0))
    -- if z < self.h then
    --     _tau =  -self.f0 * sn
    -- end

    -- BP1
    local Vi = self:Vinit(x, y)
    local sn = self:sn_pre(x, y)
    local e = math.exp((self.f0 + self.b * math.log(self.V0 / math.abs(Vi))) / self.avs)    
    local _tau = -(sn * self.avs * math.asinh((Vi / (2.0 * self.V0)) * e) + self:eta(x, y) * Vi)
    
    file = io.open ('/home/jyun/Tandem/mtmod_team3/stress_profile_'..self.pathstr,'a')
    io.output(file)
    io.write(x,'\t',y,'\t',sn,'\t',_tau,'\n')
    io.close(file)
    return _tau
end

homref = mtmod.new{sigma_vs=20,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.012,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='hom',branch_n='base'}
homref2 = mtmod.new{sigma_vs=20,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.012,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='hom',branch_n='base2'}
test = mtmod.new{sigma_vs=20,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.025,avw_asperity=0.025,Dc=5e-3,Dc_asperity=1e-2,muver='hom',branch_n='test'}
highsn_base = mtmod.new{sigma_vs=50,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.012,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='hom',branch_n='highsn'}
linear_base = mtmod.new{sigma_vs=50,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.012,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='hom',branch_n='linear_base'}
linearsn_base = mtmod.new{sigma_vs=50,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.012,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='hom',branch_n='linearsn_base'}
linearsn_tauss_base = mtmod.new{sigma_vs=50,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.012,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='hom',branch_n='linearsn_tauss_base'}
highsn_lindc_base = mtmod.new{sigma_vs=50,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.012,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='hom',branch_n='highsn_lindc_base'}
lowsn_base = mtmod.new{sigma_vs=30,sigma_asperity=30,sigma_shallow=5,avs=0.025,avw_shallow=0.012,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='hom',branch_n='lowsn'}
linearsn_shallowVS = mtmod.new{sigma_vs=50,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.02,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='hom',branch_n='linearsn_shallowVS'}
linearsn_shallowVS_locked = mtmod.new{sigma_vs=80,sigma_asperity=80,sigma_shallow=5,avs=0.025,avw_shallow=0.02,avw_asperity=0.007,Dc=5e-3,Dc_asperity=1e-2,muver='hom',branch_n='linearsn_shallowVS_locked'}
linearsn_shallowVS_locked2 = mtmod.new{sigma_vs=50,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.02,avw_asperity=0.007,Dc=5e-3,Dc_asperity=1e-2,muver='hom',branch_n='linearsn_shallowVS_locked2'}

highsn_gradmu = mtmod.new{sigma_vs=50,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.012,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='grad',branch_n='highsn_gradmu'}
highsn_gradmu2 = mtmod.new{sigma_vs=50,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.012,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='grad',branch_n='highsn_gradmu2'}
highsn_gradmu3 = mtmod.new{sigma_vs=50,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.012,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='grad',branch_n='highsn_gradmu3'}
highsn_patchmu = mtmod.new{sigma_vs=50,sigma_asperity=50,sigma_shallow=5,avs=0.025,avw_shallow=0.02,avw_asperity=0.011,Dc=5e-3,Dc_asperity=1e-2,muver='patch',branch_n='highsn_patchmu',cs_high=3.464,cs_low=2.887,mu_sed=6.8,sed_xlim=35,sed_ylim=5}