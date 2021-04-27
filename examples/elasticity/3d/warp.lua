function avocado(x, y, z)
    local r = 0.5 * (x + 1.0)
    phi = 0.5 * math.pi * y
    th = 0.5 * math.pi * z
    return r * math.cos(phi) * math.sin(th), r * math.sin(phi) * math.sin(th), r * math.cos(th)
end
