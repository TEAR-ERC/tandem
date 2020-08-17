function gravity(x, y, z)
    local rho = 1.0
    local W = 0.2
    local L = 1.0
    local delta = W / L 
    local gamma = 0.4 * delta^2
    return 0.0, 0.0, -rho * gamma
end

function left_boundary(x, y, z)
    return 0.0, 0.0, 0.0
end

function lam(x, y, z)
    return 1.25
end

function mu(x, y, z)
    return 1.0
end
