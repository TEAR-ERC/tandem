local CantileverRod = {}

-- For the cantilever rod, the purely elastic response
-- is given by the formula u_x = (\sigma_xx * L)/E
-- For the values in this file, the displacement at the
-- free end of the cantilever rod should be 5 micrometers.

function CantileverRod:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function CantileverRod:boundary(x, y, z)
    -- Fixed end for a cantilever rod
    if x < 1e-9 then
	return 0.0, 0.0, 0.0
    end
end

function CantileverRod:traction_boundary(x, y, z)
    -- 10 MPa constant stress applied as traction boundary
    -- condition on the free end of the cantilever rod
	return 0.01, 0.0, 0.0
end


function CantileverRod:mu(x, y, z)
    return 77.0 -- GPa
end

function CantileverRod:lam(x, y, z)

    return 115.0  -- GPa 
end

cantilever_rod = CantileverRod:new()
