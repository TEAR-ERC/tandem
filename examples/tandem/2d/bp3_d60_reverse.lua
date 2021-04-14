local BP3 = require "BP3"

local bp3 = BP3.new(60, 1e-9)

function boundary(x, y, t)
    return bp3:boundary(x, y, t)
end

function mu(x, y)
    return bp3:mu(x, y)
end

function lam(x, y)
    return bp3:lam(x, y)
end

function a(x, y)
    return bp3:a(x, y)
end

function eta(x, y)
    return bp3:eta(x, y)
end

function sn_pre(x, y)
    return bp3:sn_pre(x, y)
end

function tau_pre(x, y)
    return bp3:tau_pre(x, y)
end

function Vinit(x, y)
    return bp3:Vinit(x, y)
end

function Sinit(x, y)
    return bp3:Sinit(x, y)
end
